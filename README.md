```markdown
# Grammar Scoring with Whisper & Gemma (Unsloth)

This project provides an end-to-end pipeline to:

1. **Transcribe audio files** using OpenAI's Whisper.
2. **Format training data** for fine-tuning a language model on grammar score prediction.
3. **Fine-tune a Gemma-based LLM (via Unsloth)** on the prepared dataset.
4. **Evaluate grammar scores** on test data with multiple metrics.

---

## 📁 File Structure

```
├── extract_text.py              # Transcribes audio and prepares training/test datasets
├── finetune_G_eval.py          # Fine-tunes model and evaluates grammar scores
├── requirements.txt            # Required Python dependencies
├── train.csv                   # CSV with training filenames and grammar labels
├── test.csv                    # CSV with test filenames
├── train_audio/                # Folder with training audio files
├── test_audio/                 # Folder with test audio files
├── grammar_score_training_data_with_system.json  # Output JSON used for fine-tuning
├── transcribed_test_set.csv    # Output CSV used for grammar score evaluation
└── grammar_scored_test_set.csv # Final evaluated CSV with grammar scores
```

---

## 🧠 1. Audio Transcription + Dataset Generation

Run `extract_text.py` to:
- Transcribe both training and test audios using Whisper.
- Generate:
  - JSON in ShareGPT format (for model fine-tuning)
  - CSV with transcriptions (for evaluation)

### 🔧 Usage:

```bash
python extract_text.py \
  --train_csv_path ./train.csv \
  --test_csv_path ./test.csv \
  --train_audio_dir ./train_audio \
  --test_audio_dir ./test_audio \
  --output_train_json grammar_score_training_data_with_system.json \
  --output_test_csv transcribed_test_set.csv
```

### 📝 Input CSVs:

- `train.csv` should have:
  ```
  filename,label
  sample1.wav,7.5
  sample2.wav,6.0
  ```
- `test.csv` should have:
  ```
  filename
  test1.wav
  test2.wav
  ```

---

## 🧪 2. Fine-Tuning & Evaluation

Run `finetune_G_eval.py` to:
- Fine-tune a quantized `gemma-3-4b-it` model using [Unsloth](https://github.com/unslothai/unsloth).
- Use 80/10/10 split for train/eval/test.
- Evaluate predictions using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Rounded Accuracy
- Predict scores for `transcribed_test_set.csv`

### 🔧 Usage:

```bash
python finetune_G_eval.py \
  --training_json_path ./grammar_score_training_data_with_system.json \
  --input_csv_path ./transcribed_test_set.csv \
  --output_csv_path ./grammar_scored_test_set.csv \
  --eval_model True
```

---

## 📦 Installation

### ✅ Recommended: Use a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 💡 Notes

- Make sure you have GPU support (`torch.cuda.is_available()` should return `True`).
- Whisper model used: `openai/whisper-large-v3-turbo`
- Model used for fine-tuning: `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`
- If you run into CUDA memory issues, reduce batch size or model size.

---
