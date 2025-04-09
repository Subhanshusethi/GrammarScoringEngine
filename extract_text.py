#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
from tqdm import tqdm
import json
import pandas as pd
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import argparse

def load_whisper_model():
    """
    Load and configure the Whisper model for automatic speech recognition.

    Returns:
        pipeline: Configured ASR pipeline.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True
        )
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {e}")

def process_audio_dataset(
    train_csv_path,
    test_csv_path,
    train_audio_dir,
    test_audio_dir,
    output_train_json="grammar_score_training_data_with_system.json",
    output_test_csv="transcribed_test_set.csv"
):
    """
    Processes training and test datasets:
    - Transcribes audio files using ASR pipeline.
    - Formats training data in ShareGPT format with grammar score.
    - Transcribes test audio and outputs CSV.

    Args:
        train_csv_path (str): Path to training CSV with 'filename' and 'label' columns.
        test_csv_path (str): Path to test CSV with 'filename' column.
        train_audio_dir (str): Directory with training audio files.
        test_audio_dir (str): Directory with test audio files.
        output_train_json (str): Output JSON path for formatted training data (with system role).
        output_test_csv (str): Output CSV path for test audio transcriptions.

    Raises:
        ValueError: If input CSVs lack required columns.
        RuntimeError: If file I/O or transcription fails critically.
    """
    # Load the Whisper pipeline
    pipe = load_whisper_model()

    # Validate input CSVs
    train_df = pd.read_csv(train_csv_path)
    if "filename" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train_csv_path must contain 'filename' and 'label' columns")
    
    test_df = pd.read_csv(test_csv_path)
    if "filename" not in test_df.columns:
        raise ValueError("test_csv_path must contain 'filename' column")

    # ========= Train Dataset Processing ========= #
    print("\nProcessing training dataset...")
    formatted_train_data = []

    for filename in tqdm(os.listdir(train_audio_dir), desc="Transcribing training audios"):
        audio_path = os.path.join(train_audio_dir, filename)
        score_row = train_df[train_df["filename"] == filename]

        if not score_row.empty:
            try:
                result = pipe(audio_path)
                text = result["text"]
                score = score_row["label"].values[0]
                sample = {
                    "messages": [
                        {"role": "system", "content": "You are an assistant"},
                        {"role": "user", "content": f"Evaluate the grammar score for the following text:\n\"{text}\""},
                        {"role": "assistant", "content": f"The score for the given text is {score}"}
                    ]
                }
                formatted_train_data.append(sample)
            except Exception as e:
                print(f"Error transcribing {filename}: {e}")
        else:
            print(f"Label not found for file: {filename}")

    try:
        with open(output_train_json, "w") as f:
            json.dump(formatted_train_data, f, indent=4)
        print(f"Training data with system role saved to: {output_train_json}")
    except Exception as e:
        raise RuntimeError(f"Error writing to {output_train_json}: {e}")

    # ========= Test Dataset Processing ========= #
    print("\nProcessing test dataset...")
    transcribed_results = []

    for filename in tqdm(test_df["filename"], desc="Transcribing test audios"):
        audio_path = os.path.join(test_audio_dir, filename)
        if os.path.exists(audio_path):
            try:
                result = pipe(audio_path)
                text = result["text"]
                transcribed_results.append({"filename": filename, "transcribed_text": text})
            except Exception as e:
                print(f"Error transcribing {filename}: {e}")
        else:
            print(f"Audio file not found: {filename}")

    try:
        pd.DataFrame(transcribed_results).to_csv(output_test_csv, index=False)
        print(f"Test transcriptions saved to: {output_test_csv}")
    except Exception as e:
        raise RuntimeError(f"Error writing to {output_test_csv}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio dataset for grammar scoring.")
    parser.add_argument("--train_csv_path", type=str, required=True, help="Path to training CSV with labels.")
    parser.add_argument("--test_csv_path", type=str, required=True, help="Path to test CSV with filenames.")
    parser.add_argument("--train_audio_dir", type=str, required=True, help="Directory with training audio files.")
    parser.add_argument("--test_audio_dir", type=str, required=True, help="Directory with test audio files.")
    parser.add_argument("--output_train_json", type=str, default="grammar_score_training_data_with_system.json",
                        help="Output JSON path for training data.")
    parser.add_argument("--output_test_csv", type=str, default="transcribed_test_set.csv",
                        help="Output CSV path for test transcriptions.")
    args = parser.parse_args()

    try:
        process_audio_dataset(
            args.train_csv_path,
            args.test_csv_path,
            args.train_audio_dir,
            args.test_audio_dir,
            args.output_train_json,
            args.output_test_csv
        )
    except Exception as e:
        print(f"Failed to process audio dataset: {e}")
        exit(1)