# 🔍 Metaphor Detection using DistilBERT

This project uses a fine-tuned DistilBERT model to **detect metaphorical usage** of specific words in a sentence. It focuses on identifying whether a metaphorical sense is present based on a labeled dataset and predicts the usage as either *literal* or *metaphorical*.

---

## 🧠 Project Overview

Given a paragraph and a metaphor word (e.g., "road", "candle"), this system:

1. Extracts the sentence containing that metaphor.
2. Determines whether the usage is **metaphorical** or **literal**.
3. Fine-tunes a transformer model (DistilBERT) using these labeled examples for binary classification.

---

## 🚀 Features

- Extracts relevant sentences containing metaphor words
- Fine-tunes `DistilBERT` for binary classification
- Uses metaphor-specific sentence pairs for better contextual understanding
- Implements early stopping to prevent overfitting
- Provides classification reports and saves predictions
- Saves trained model to disk

---

## 🧾 Dataset Format

The CSV input must contain at least the following columns:

| text | metaphorID | label_boolean |
|------|-------------|----------------|
| Long paragraph... | road | 1 |

- `text`: Full paragraph or multiple sentences
- `metaphorID`: One of the target metaphor words (`road`, `candle`, `light`, `spice`, `ride`, `train`, `boat`)
- `label_boolean`: `1` for metaphorical use, `0` for literal

---

## 🛠️ Setup

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/metaphor-detection-distilbert.git
cd metaphor-detection-distilbert
