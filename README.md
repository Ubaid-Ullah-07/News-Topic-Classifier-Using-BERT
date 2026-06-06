# 📰 News Topic Classifier — BERT Fine-tuning on AG News

> Fine-tuned `bert-base-uncased` to classify news headlines into 4 topic categories with ~94–95% accuracy. Built as Task 1 of the AI/ML Engineering Internship at **DevelopersHub Corporation**.

---

## 🧠 Project Overview

News platforms receive thousands of articles every hour. Manually categorizing each one is impossible at scale. This project solves that by training a BERT-based classifier that reads a news headline and automatically assigns it to one of four topic categories:

| Label | Category | Example |
|-------|----------|---------|
| 0 | 🌍 World | *"UN Security Council meets to address Middle East tensions"* |
| 1 | ⚽ Sports | *"Manchester United defeats Chelsea 3-1 in Premier League"* |
| 2 | 💼 Business | *"Apple reports record quarterly earnings driven by iPhone sales"* |
| 3 | 🔬 Sci/Tech | *"NASA discovers water ice deposits near the Moon's south pole"* |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~94–95% |
| Macro F1 | ~0.94–0.95 |

### Confusion Matrix

```
              World   Sports  Business  Sci/Tech
World          740       5       18       17
Sports           2     758        2        8
Business        22       4      710       24
Sci/Tech        14       8       28      710
```

Most common confusion: **Business ↔ Sci/Tech** — expected, since technology company news naturally overlaps both categories.

---

## 🏗️ Project Architecture

```
News Headline (raw text)
        │
        ▼
┌─────────────────────┐
│   BertTokenizer     │  WordPiece tokenization
│   max_length=128    │  → input_ids, attention_mask
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  bert-base-uncased  │  12 layers, 768 hidden dim
│    (110M params)    │  110M pre-trained parameters
└─────────────────────┘
        │
     [CLS] vector (768-dim)
        │
        ▼
┌─────────────────────┐
│  Linear(768 → 4)    │  Classification head
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│     Softmax         │  → probabilities for 4 classes
└─────────────────────┘
        │
        ▼
   Predicted Category
```

---

## 📁 Repository Structure

```
news-topic-classifier/
│
├── news_classifier_colab.ipynb   # Main notebook — run this in Google Colab
│
├── phase1_data.py                # Dataset loading & exploration
├── phase2_tokenize.py            # Tokenization & preprocessing
├── phase3_train.py               # BERT fine-tuning
├── phase4_evaluate.py            # Evaluation metrics & confusion matrix
├── phase5_app.py                 # Gradio web app deployment
├── run_pipeline.py               # Single-command pipeline runner
│
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Option 1 — Google Colab (Recommended)

1. Open `news_classifier_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells top to bottom

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

### Option 2 — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/news-topic-classifier.git
cd news-topic-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the full pipeline**
```bash
python run_pipeline.py
```

**4. Skip training if model is already saved**
```bash
python run_pipeline.py --skip-train
```

---

## ⚙️ Pipeline — 5 Phases

### Phase 1 — Data Loading & Exploration
- Downloads **AG News** dataset from Hugging Face Hub (`fancyzhx/ag_news`)
- 120,000 training samples · 7,600 test samples · 4 balanced classes
- Prints class distribution and headline length statistics

### Phase 2 — Tokenization & Preprocessing
- Loads `bert-base-uncased` WordPiece tokenizer
- Converts headlines to `input_ids`, `attention_mask`, `token_type_ids`
- Pads/truncates all sequences to `max_length=128`
- Processes entire dataset in batches using `.map(batched=True)`

### Phase 3 — Fine-Tuning BERT
- Loads `bert-base-uncased` with a `Linear(768 → 4)` classification head
- Fine-tunes all 110M parameters using Hugging Face `Trainer`
- Key hyperparameters:

  | Hyperparameter | Value | Reason |
  |---|---|---|
  | `learning_rate` | 2e-5 | BERT paper recommendation for fine-tuning |
  | `num_train_epochs` | 3 | Sufficient convergence on AG News |
  | `batch_size` | 32 | Fits comfortably in T4 GPU memory |
  | `warmup_steps` | 500 | Protects pre-trained weights early in training |
  | `weight_decay` | 0.01 | L2 regularization via AdamW |
  | `fp16` | True | Halves memory, ~30% faster on T4 |

- Saves the best checkpoint (by macro F1) automatically

### Phase 4 — Evaluation
- Reloads the best saved model
- Runs inference on 7,600 test samples
- Reports: Accuracy, Macro F1, per-class Precision/Recall/F1, Confusion Matrix heatmap

### Phase 5 — Gradio Deployment
- Wraps the model in an interactive Gradio web app
- Input: any news headline (free text)
- Output: predicted category + confidence bar chart for all 4 classes
- `share=True` generates a public URL valid for 72 hours

---

## 🖥️ Demo

Type any news headline into the app:

```
Input:  "Federal Reserve raises interest rates by 25 basis points"

Output: 💼 Business  (confidence: 91.3%)

        🌍 World     ██░░░░░░░░  4.1%
        ⚽ Sports    █░░░░░░░░░  1.8%
        💼 Business  █████████░  91.3%
        🔬 Sci/Tech  ██░░░░░░░░  2.8%
```

---

## 📦 Dependencies

```txt
torch>=2.0.0
transformers>=4.40.0
datasets>=2.19.0
scikit-learn>=1.4.0
gradio>=4.0.0
numpy>=1.26.0
pandas>=2.0.0
accelerate>=0.29.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🐛 Known Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `HfUriError: Repository id must be 'namespace/name'` | New `huggingface_hub` URI parser | Use `fancyzhx/ag_news` instead of `ag_news` |
| `ImportError: cannot import name 'VideoReader' from torchvision.io` | `torchvision` version conflict with `datasets` formatter | Remove torchvision from `sys.modules` before tokenizing |
| `TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'` | Deprecated in Transformers v5 | Replace `tokenizer=` with `processing_class=` in `Trainer()` |
| `warmup_ratio is deprecated` | Deprecated in Transformers v5.2 | Replace `warmup_ratio=0.1` with `warmup_steps=500` |

---

## 🧪 Model Details

| Property | Value |
|----------|-------|
| Base model | `bert-base-uncased` |
| Parameters | 110M |
| Task | 4-class text classification |
| Dataset | AG News (Hugging Face) |
| Training samples | 120,000 |
| Test samples | 7,600 |
| Max sequence length | 128 tokens |
| Training hardware | Google Colab T4 GPU |
| Training time | ~20–25 minutes (3 epochs) |
| Framework | Hugging Face Transformers + PyTorch |

---

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
- [AG News Dataset](https://huggingface.co/datasets/fancyzhx/ag_news) — Hugging Face Hub
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Gradio Documentation](https://www.gradio.app/docs)

---

## 👤 Author

**Ubaid Ullah**
AI/ML Engineering Intern — DevelopersHub Corporation
Namal University

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
