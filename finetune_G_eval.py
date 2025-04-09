#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import pandas as pd
from tqdm import tqdm
from torch.cuda import is_available as is_torch_available
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def fine_tune_model(training_json_path):
    # Load model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # Load and split dataset
    with open(training_json_path, 'r') as file:
        dataset = json.load(file)

    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Convert to HuggingFace Datasets
    def prepare(ds):
        ds = standardize_data_formats(ds)
        ds = Dataset.from_list(ds)
        ds = ds.map(lambda examples: {
            "text": [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in examples["messages"]]
        }, batched=True)
        return ds

    train_dataset = prepare(train_data)
    eval_dataset = prepare(val_data)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            report_to="none",
            evaluation_strategy="no",
            save_strategy="no",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    trainer.train()

    return model, tokenizer, test_data


def evaluate_grammar_scores(model, tokenizer, input_csv_path, output_csv_path="grammar_scored_test_set.csv",
                            chat_template="gemma-3", max_new_tokens=64, temperature=0.7, top_p=0.95, top_k=64,
                            device="cuda" if is_torch_available() else "cpu"):
    df = pd.read_csv(input_csv_path)
    labels = []

    for text in tqdm(df["transcribed_text"], desc="Evaluating"):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": f"Evaluate the grammar score for the following text:\n\"{text}\""}]}
        ]
        chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer([chat_text], return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        response = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        try:
            score = float("".join(filter(lambda c: c.isdigit() or c == ".", response.split()[-1])))
        except:
            score = -1

        labels.append(score)

    df["label"] = labels
    df.drop(columns=["transcribed_text"], inplace=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Done! Saved to {output_csv_path}")


def evaluate_on_test_set(model, tokenizer, test_data, device):
    true_scores = []
    pred_scores = []

    for example in tqdm(test_data, desc="Evaluating held-out test set"):
        messages = example["messages"]
        prompt_text = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
        true_score = float(messages[-1]["content"].split()[-1])

        inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7, top_p=0.95, top_k=64)
        response = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        try:
            pred_score = float("".join(filter(lambda c: c.isdigit() or c == ".", response.split()[-1])))
        except:
            pred_score = -1.0

        true_scores.append(true_score)
        pred_scores.append(pred_score)

    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = mean_squared_error(true_scores, pred_scores, squared=False)
    accuracy = sum(1 for t, p in zip(true_scores, pred_scores) if round(t) == round(p)) / len(true_scores)

    print("\nHeld-out Test Set Evaluation Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Rounded Accuracy: {accuracy:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model and evaluate grammar scores.")
    parser.add_argument("--training_json_path", type=str, default="./grammar_score_training_data_with_system.json",
                        help="Path to the training JSON file.")
    parser.add_argument("--input_csv_path", type=str, default="./transcribed_test_set.csv",
                        help="Path to the input CSV file with transcribed text.")
    parser.add_argument("--output_csv_path", type=str, default="./grammar_scored_test_set.csv",
                        help="Path to save the output CSV file with grammar scores.")
    parser.add_argument("--eval_model", type=bool, default=True,
                        help="Whether to evaluate the model after fine-tuning.")
    args = parser.parse_args()

    print("Fine-tuning the model...")
    model, tokenizer, test_data = fine_tune_model(args.training_json_path)

    if args.eval_model:
        print("\nEvaluating on transcribed test set...")
        evaluate_grammar_scores(
            model=model,
            tokenizer=tokenizer,
            input_csv_path=args.input_csv_path,
            output_csv_path=args.output_csv_path,
            chat_template="gemma-3",
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            device="cuda" if is_torch_available() else "cpu"
        )

        print("\nEvaluating on held-out test split...")
        evaluate_on_test_set(
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            device="cuda" if is_torch_available() else "cpu"
        )


if __name__ == "__main__":
    main()
