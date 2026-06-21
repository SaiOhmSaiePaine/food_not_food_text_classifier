# 🍗🚫🥑 Food Not Food Text Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-latest-ffd21e?logoColor=white)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-6e8b3d)
![Gradio](https://img.shields.io/badge/Demo-Gradio-orange?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-green)
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

A fine-tuned **DistilBERT** text classification model that classifies a sentence as either **"food"** or **"not food"** — from a single line of text to a batch of thousands.

---

## 📖 Description

Deciding whether a piece of text is *about food* is harder than it looks. Metaphors ("let's marinate these ideas"), slang ("couch potato"), and food words used in non-food contexts ("cookies" in a browser, "raspberry pi") easily trip up keyword-based systems.

This project solves that with a transformer-based classifier. It fine-tunes [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) (~67M parameters) on a small synthetic dataset of 250 food/not-food image captions, achieving high accuracy on a held-out test set. The trained model is published to the Hugging Face Hub and served through an interactive Gradio web demo.

**Live model:** [`SaiePaine/learn_hf_food_not_food_text_classifier-distilbert-base-uncased`](https://huggingface.co/SaiePaine/learn_hf_food_not_food_text_classifier-distilbert-base-uncased)

---

## 🏗️ Architecture / Workflow

The project follows a complete end-to-end NLP pipeline across three notebooks:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA                                           │
│   mrdbourke/learn_hf_food_not_food_image_captions  (250 captions)       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. PREPROCESS                                                          │
│     • Map string labels → {0: not_food, 1: food}                        │
│     • Train/test split (80/20, seed=42)                                 │
│     • Tokenize with DistilBERT AutoTokenizer (padding + truncation)     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. TRAIN                                                               │
│     • AutoModelForSequenceClassification (2-class head)                 │
│     • Trainer + TrainingArguments                                       │
│       (lr=1e-4, batch=32, epochs=10, eval/save per epoch)               │
│     • Metric: accuracy via 🤗 Evaluate                                  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. EVALUATE & SAVE                                                     │
│     • Predict on test set, compute accuracy                             │
│     • Inspect low-confidence predictions for insight                    │
│     • Save locally + push to Hugging Face Hub                           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. SERVE                                                               │
│     • Gradio web interface (text in → label probabilities out)          │
│     • Deployable to Hugging Face Spaces                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Notebooks

| Notebook | Purpose |
|----------|---------|
| [`huggingface_text_classification_tutorial.ipynb`](huggingface_text_classification_tutorial.ipynb) | Full learning walkthrough: data loading → tokenization → training → evaluation → inference (pipeline + PyTorch modes). |
| [`Food_not_Food_Text_Classification_Customized_Model.ipynb`](Food_not_Food_Text_Classification_Customized_Model.ipynb) | Consolidated, production-style training script that trains and saves the model. |
| [`Demo.ipynb`](Demo.ipynb) | Loads the published model and wraps it in an interactive Gradio demo; deploys to Hugging Face Spaces. |

---

## ✅ Prerequisites

- **Python 3.10+**
- A **GPU** is strongly recommended for training (Google Colab T4 works well). CPU works for inference only.
- A free [Hugging Face account](https://huggingface.co) (optional — only needed to download the dataset, push the model, or deploy the demo).

### Python dependencies

```
torch
transformers
datasets
evaluate
accelerate
gradio
numpy
pandas
scikit-learn
matplotlib
```

---

## ⚙️ Setup and Installation

```bash
# 1. Clone the repository
git clone https://github.com/SaiOhmSaiePaine/food_not_food_text_classifier.git
cd food_not_food_text_classifier

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -U pip
pip install torch transformers datasets evaluate accelerate gradio numpy pandas scikit-learn matplotlib

# 4. (Optional) Authenticate with Hugging Face for push/deploy steps
huggingface-cli login
```

> **Easiest path:** Open any notebook in Google Colab (use the Colab badge above) — it auto-installs missing packages and provides a free T4 GPU.

---

## 🚀 Usage

### Option A — Use the published model (no training required)

The fine-tuned model is already on the Hugging Face Hub. Load it directly with the `transformers` pipeline:

```python
import torch
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="SaiePaine/learn_hf_food_not_food_text_classifier-distilbert-base-uncased",
    device="cuda" if torch.cuda.is_available() else "cpu",
    top_k=None,
    batch_size=32,
)

# Single prediction
result = classifier("A delicious plate of scrambled eggs, bacon and toast.")
print(result)
# [[{'label': 'food',     'score': 0.9987},
#   {'label': 'not_food', 'score': 0.0013}]]

# Batch prediction
classifier([
    "Sizzling pepperoni pizza with melted mozzarella.",
    "My Wi-Fi connection keeps dropping during the meeting.",
])
```

### Option B — Train the model yourself

Run the training notebook end-to-end:

```bash
jupyter notebook Food_not_Food_Text_Classification_Customized_Model.ipynb
```

Or open it in Colab via the badge at the top of the notebook. The trained model is saved to `models/learn_hf_food_not_food_text_classifier-distilbert-base-uncased/`.

### Option C — Launch the Gradio demo

```bash
jupyter notebook Demo.ipynb
```

Run the cells to spin up a local Gradio web UI where you can type any sentence and see the food / not-food probabilities. The last cells also publish the demo to a Hugging Face Space.

### Inference with PyTorch directly (no pipeline)

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "SaiePaine/learn_hf_food_not_food_text_classifier-distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "A bowl of spicy lamb rogan josh with basmati rice."
inputs = tokenizer(text, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)

pred_id = outputs.logits.argmax().item()
prob = torch.softmax(outputs.logits, dim=1).max().item()
print(f"Label: {model.config.id2label[pred_id]}  (confidence: {prob:.4f})")
```

---

## 🛠️ Technologies Used

- **[PyTorch](https://pytorch.org/)** — tensor computation and the deep-learning backend
- **[🤗 Transformers](https://huggingface.co/docs/transformers)** — model loading (`AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `pipeline`)
- **[🤗 Datasets](https://huggingface.co/docs/datasets)** — loading and preprocessing the caption dataset
- **[🤗 Evaluate](https://huggingface.co/docs/evaluate)** — accuracy metric computation
- **[Accelerate](https://huggingface.co/docs/accelerate)** — distributed/mixed-precision training support
- **[DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)** — the fine-tuned transformer backbone (~67M params)
- **[Gradio](https://gradio.app/)** — interactive web demo UI
- **scikit-learn / pandas / NumPy / matplotlib** — evaluation, data handling, and loss plotting
- **Google Colab** — GPU-backed development environment

---

## 📄 License

Distributed under the **Apache-2.0** License. See [`LICENSE`](LICENSE) for details.
