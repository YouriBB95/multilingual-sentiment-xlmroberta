# 🌍 Multilingual Sentiment Analysis with XLM-RoBERTa

Fine-tuning `xlm-roberta-base` to classify customer reviews in **English and Spanish** as positive or negative using Hugging Face Transformers.

---

## 🎯 Goal

Build a multilingual sentiment classifier that generalizes across languages using real Amazon review data.

---

## 🗂️ Dataset

- **Source**: [`clapAI/MultiLingualSentiment`](https://huggingface.co/datasets/clapAI/MultiLingualSentiment)
- Languages used: 🇬🇧 English, 🇪🇸 Spanish
- Filtered out neutral sentiment → only binary classification (`positive` = 1, `negative` = 0)
- Sampled 6,000 reviews (80% train / 20% eval)

---

## 🧠 Model

- **Base model**: `xlm-roberta-base` (cross-lingual pretrained transformer by Facebook AI)
- Fine-tuned for **binary classification**
- Tokenized with Hugging Face `XLMRobertaTokenizer`

---

## 🏋️‍♀️ Training

- 2 epochs on 4,800 training samples
- Batch size: 8
- Evaluation every epoch
- Optimized using Hugging Face `Trainer`

```python
TrainOutput(
  global_step=600,
  training_loss=0.3979,
  metrics={
    'train_runtime': 717s,
    'train_samples_per_second': 13.4,
    'train_loss': 0.3979,
    'epoch': 2.0
  }
)
