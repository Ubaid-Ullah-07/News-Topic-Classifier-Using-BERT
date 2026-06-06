# 📰 AG News Topic Classifier using BERT

## Overview
This project fine-tunes **BERT (bert-base-uncased)** on the **AG News** dataset to classify news headlines into four categories:

- 🌍 World
- ⚽ Sports
- 💼 Business
- 🔬 Sci/Tech

The notebook covers the complete machine learning workflow:
1. Dataset loading and exploration
2. Text preprocessing and tokenization
3. BERT fine-tuning
4. Model evaluation
5. Gradio web application deployment

---

## Project Structure

```text
news_classifier_colab.ipynb
README.md
saved_model/
```

---

## Dataset

**AG News Dataset**
- Source: Hugging Face Datasets
- Training samples: 120,000
- Test samples: 7,600
- Classes: 4

| Label | Category |
|---------|---------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

---

## Model

- Base Model: `bert-base-uncased`
- Architecture:
  - BERT Encoder (110M parameters)
  - Linear Classification Head (4 outputs)

All BERT layers are fine-tuned.

---

## Installation

```bash
pip install transformers datasets accelerate gradio scikit-learn
```

---

## Training Configuration

| Parameter | Value |
|------------|--------|
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Train Batch Size | 32 |
| Eval Batch Size | 64 |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |

---

## Tokenization

The model uses the BERT tokenizer:

```python
BertTokenizer.from_pretrained("bert-base-uncased")
```

Generated features:

- `input_ids`
- `attention_mask`
- `token_type_ids`

---

## Evaluation

Metrics:
- Accuracy
- Macro F1 Score
- Classification Report
- Confusion Matrix

Example confusion matrix results from the trained model:

| True \\ Pred | World | Sports | Business | Sci/Tech |
|-------------|-------|--------|----------|----------|
| World | 1809 | 9 | 40 | 42 |
| Sports | 10 | 1876 | 7 | 7 |
| Business | 31 | 4 | 1743 | 122 |
| Sci/Tech | 20 | 10 | 96 | 1774 |

Observations:
- Sports is classified with very high accuracy.
- Most confusion occurs between Business and Sci/Tech.
- World and Business occasionally overlap due to economic/political news.

---

## Running Training

Execute all notebook cells sequentially:

```python
trainer.train()
```

The best model is automatically saved:

```python
trainer.save_model("saved_model")
tokenizer.save_pretrained("saved_model")
```

---

## Gradio Web Application

Launch the interface:

```python
gr.Interface(...).launch(share=True)
```

Features:
- Headline classification
- Confidence scores for all classes
- Shareable public link

Example input:

```text
Apple unveils new AI-powered iPhone features
```

Output:

```text
Sci/Tech
```

---

## Results

The fine-tuned BERT model achieves strong performance on AG News and provides reliable predictions across all four categories.

Key strengths:
- High classification accuracy
- Fast inference
- Easy deployment using Gradio
- End-to-end NLP workflow

---

## Future Improvements

- Hyperparameter tuning
- DistilBERT for faster inference
- Model quantization
- Deployment on Hugging Face Spaces
- Error analysis for Business vs Sci/Tech samples

---

## Author
**Ubaid Ullah** AI/ML Engineering Intern — DevelopersHub Corporation


Developed as an NLP text-classification project using Hugging Face Transformers and BERT.
