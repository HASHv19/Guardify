# 🛡️ Guardify – Beginner's Complete Guide


## 📚 Table of Contents

1. [What is Guardify?](#what-is-guardify)
2. [Project Structure](#project-structure)
3. [Key Concepts](#key-concepts)
4. [How the System Works End-to-End](#how-the-system-works-end-to-end)
5. [Understanding Each Component](#understanding-each-component)
6. [Step-by-Step: Running the Project](#step-by-step-running-the-project)
7. [Common Issues & Solutions](#common-issues--solutions)
8. [Glossary](#glossary)

---

## 🎯 What is Guardify?

Guardify is a **bilingual cyberbullying detection system** that classifies social media posts as **Bullying** or **Non-Bullying**. It is special because it handles both:

- **English**: `"You are such a loser, get lost"`
- **Hinglish** (Hindi + English mixed): `"Tu kitna useless banda hai bro"`
- **Transliterated Hindi**: `"Tumhara attitude bohot cheap hai"`

Most existing systems only work on pure English. Guardify is built specifically for the reality of Indian social media, where people switch languages mid-sentence.

---

## 📁 Project Structure

```
Guardify/
├── apps/
│   ├── api/
│   │   └── main.py              ← FastAPI backend (the server)
│   └── web/
│       └── package.json         ← React frontend (the UI)
│
├── configs/
│   ├── baseline.yaml            ← Config for SVM/LR training
│   └── muril.yaml               ← Config for MuRIL transformer training
│
├── data/
│   ├── raw/
│   │   └── data.csv             ← Small demo fixture (default configs use data/external/*.csv)
│   └── external/                ← Drop HASOC, TRAC datasets here
│
├── src/
│   ├── guardify/                ← The core Python package (all real logic)
│   │   ├── config.py            ← Loads and merges YAML config files
│   │   ├── train.py             ← Training entrypoint (CLI)
│   │   ├── predict.py           ← Prediction entrypoint (CLI)
│   │   ├── evaluate.py          ← Evaluation entrypoint (CLI)
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── pipeline.py      ← Text cleaning logic (the main preprocessor)
│   │   │   └── lexicon.py       ← Emoji map, abuse words, obfuscation patterns
│   │   │
│   │   ├── data/
│   │   │   ├── loaders.py       ← Loads HASOC, TRAC, CSV, JSON datasets
│   │   │   └── schema.py        ← Converts all datasets to a standard format
│   │   │
│   │   ├── features/
│   │   │   └── tfidf.py         ← TF-IDF vectorizer builder
│   │   │
│   │   ├── models/
│   │   │   ├── baseline.py      ← SVM and Logistic Regression definitions
│   │   │   └── transformer.py   ← MuRIL model architecture
│   │   │
│   │   ├── training/
│   │   │   ├── baseline.py      ← Full training workflow for SVM/LR
│   │   │   ├── transformer.py   ← Full training workflow for MuRIL
│   │   │   └── artifacts.py     ← Saves trained models as bundles
│   │   │
│   │   ├── inference/
│   │   │   └── service.py       ← Loads a bundle and runs predictions
│   │   │
│   │   └── evaluation/
│   │       └── metrics.py       ← Accuracy, F1, confusion matrix, etc.
│   │
│   ├── dataset.py               ← Legacy wrapper (proxies to guardify package)
│   ├── preprocess.py            ← Legacy wrapper
│   ├── model.py                 ← Legacy wrapper
│   ├── train.py                 ← Legacy wrapper
│   └── predict.py               ← Legacy wrapper
│
├── artifacts/                   ← Trained model bundles saved here
│   └── baseline/
│       └── sample-baseline/     ← Example trained bundle
│           ├── estimator.joblib ← Saved SVM/LR model
│           ├── metadata.json    ← Model info and metrics
│           └── metrics.json     ← Evaluation results
│
├── tests/                       ← Automated tests
├── example_workflow.py          ← Quick demo script
├── setup_test.py                ← Environment check
└── requirements.txt             ← Python dependencies
```

> **Important**: The files inside `src/guardify/` are the real code. The files directly in `src/` (dataset.py, preprocess.py etc.) are old wrappers that just call into the guardify package. Always look inside `src/guardify/` to understand what actually runs.

---

## 🧠 Key Concepts

### Machine Learning (ML)
Teaching a computer to recognize patterns by showing it labeled examples. You show it thousands of messages labeled "Bullying" or "Non-Bullying" and it learns to tell them apart.

### TF-IDF (Term Frequency – Inverse Document Frequency)
A way to convert text into numbers. It measures how important a word is in a document compared to across all documents. Abusive words like "idiot", "useless", "cheap" appear much more in bullying posts — TF-IDF captures this statistically.

### SVM (Support Vector Machine)
A classical ML classifier. It finds the best boundary to separate bullying from non-bullying text. Works very well with TF-IDF because TF-IDF creates high-dimensional sparse vectors, which is exactly where SVM excels.

### Transformer Models (MuRIL)
A deep learning architecture that reads entire sentences at once and understands context. MuRIL (Multilingual Representations for Indian Languages) is a transformer built by Google specifically for Indian languages including Hinglish. Unlike SVM, it understands that "cheap" in "Tumhara attitude bohot cheap hai" is being used aggressively based on the surrounding words.

### Preprocessing
Cleaning raw noisy social media text before feeding it to any model. Social media posts have URLs, @mentions, emojis, misspellings, obfuscation (b***h), and slang — preprocessing normalizes all of this.

### Model Bundle
After training, Guardify saves everything needed to make predictions into a folder called a **bundle**. This includes the trained model weights, the vectorizer, label mapping, and metadata. The API loads a bundle to serve predictions.

### Tokenization
Breaking text into smaller units (tokens) that the model processes. MuRIL uses subword tokenization — it breaks "kameena" into sub-pieces — which helps it handle spelling variations in Hinglish.

---

## ⚙️ How the System Works End-to-End

```
Raw Text Input
      │
      ▼
┌─────────────────────────────────┐
│  Preprocessing Pipeline         │
│  preprocessing/pipeline.py      │
│                                 │
│  1. Replace emojis → [ANGRY]    │
│  2. Remove URLs, @mentions      │
│  3. Strip # from hashtags       │
│  4. Fix obfuscation (b***h)     │
│  5. Collapse repeated chars     │
│  6. Lowercase                   │
│  7. Flag abuse lexicon tokens   │
└─────────────┬───────────────────┘
              │ normalized_text + flagged_tokens
              ▼
┌─────────────────────────────────────────────────┐
│  Model Layer (two paths)                        │
│                                                 │
│  PATH A - Baseline (SVM/LR)                     │
│  features/tfidf.py → TF-IDF vectorizer          │
│  models/baseline.py → SVM or Logistic Reg.      │
│                                                 │
│  PATH B - Transformer (MuRIL)                   │
│  models/transformer.py → MuRIL + classifier     │
│  Tokenizer → attention → softmax output         │
└─────────────┬───────────────────────────────────┘
              │ label + confidence + probabilities
              ▼
┌─────────────────────────────────┐
│  Inference Service              │
│  inference/service.py           │
│  Loads bundle, runs prediction  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  FastAPI Backend                │
│  apps/api/main.py               │
│  POST /predict                  │
└─────────────┬───────────────────┘
              │ JSON response
              ▼
┌─────────────────────────────────┐
│  React Frontend                 │
│  apps/web/                      │
│  User sees result in browser    │
└─────────────────────────────────┘
```

---

## 🔍 Understanding Each Component

### `preprocessing/pipeline.py` – The Text Cleaner

This runs every piece of text through a fixed sequence of steps in this exact order:

1. **Emoji replacement** — `😡` becomes `[ANGRY]`, `😂` becomes `[LAUGH]`
2. **Optional transliteration** — Roman Hindi to Devanagari (disabled by default)
3. **URL removal** — `https://example.com` → removed
4. **@mention removal** — `@username` → removed
5. **Hashtag cleanup** — `#loser` → `loser` (keeps the word, removes the #)
6. **Obfuscation fixing** — `b***h`, `b.i.t.c.h` → `bitch` using regex patterns
7. **Repeated character collapsing** — `stuuuuupid` → `stupid`
8. **Lowercase**
9. **Abuse lexicon flagging** — `cheap`, `idiot`, `chutiya` → `[ABUSIVE]`

Output is a `PreprocessResult` object with three fields: `original_text`, `normalized_text`, and `flagged_tokens`.

---

### `preprocessing/lexicon.py` – Static Word Lists

Three dictionaries used by the pipeline:

- **`EMOJI_MAP`** — maps emoji characters to text tokens (`😡` → `[ANGRY]`)
- **`ABUSE_LEXICON`** — maps abusive words to `[ABUSIVE]` placeholder; also records which tokens were flagged
- **`OBFUSCATION_PATTERNS`** — regex patterns that catch letter-substituted abuse (`b[\W_]*i[\W_]*t[\W_]*c[\W_]*h`)

---

### `data/loaders.py` – Dataset Loader

Reads datasets from CSV, Excel, or JSON and passes them to `schema.py` for standardization. Supports these dataset types out of the box:

| Type | Expected columns | Use for |
|---|---|---|
| `sample_csv` | `text`, `label` | Demo fixture data |
| `hasoc_csv` | `text`, `task_1` | HASOC shared task data |
| `trac_csv` | `text`, `label` | TRAC dataset |
| `custom_csv` | `text`, `label` | Any new CSV you add |
| `custom_json` | `text`, `label` | Any JSON dataset |

Multiple datasets are combined and deduplicated automatically.

---

### `data/schema.py` – The Canonical Format

No matter which dataset you load, it gets converted into this standard structure:

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `text` | Raw post text |
| `original_label` | Label as it came from the source file |
| `binary_label` | Standardized: `"Bullying"` or `"Non-Bullying"` |
| `source` | Which dataset it came from |
| `language_hint` | Auto-detected: `latin`, `hindi`, `mixed`, `urdu`, `unknown` |
| `split` | `train`/`test` if specified, otherwise `unspecified` |

Also automatically detects language by checking if text contains Devanagari characters, Latin characters, or both.

---

### `training/baseline.py` – SVM / Logistic Regression Training

What happens when you run baseline training:

1. Loads and preprocesses the dataset
2. Splits are **config-driven** using `training.test_size` and `training.val_size` (with current defaults: **train (50%) / validation (25%) / test (25%)**) using stratified splitting (preserving class balance in each split)
3. Fits TF-IDF vectorizer on training text
4. Trains both Logistic Regression and Linear SVM
5. Picks the best model based on validation F1 score
6. Evaluates the winner on the test set
7. Optionally runs k-fold cross validation
8. Saves the bundle to `artifacts/baseline/<output_name>/`

Bundle contains: `estimator.joblib` (vectorizer + model), `metrics.json`, `metadata.json`, `misclassified.csv`

---

### `training/transformer.py` – MuRIL Training

What happens when you run transformer training:

1. Loads and preprocesses the dataset
2. Splits into train / validation / test
3. Downloads MuRIL from HuggingFace (first time: ~500MB)
4. Creates `TextDataset` objects — tokenizes each post and returns `input_ids` + `attention_mask` tensors
5. Trains using **AdamW optimizer** with a **linear learning rate schedule**
6. After each epoch, evaluates on validation set and saves the best weights
7. Final evaluation on test set
8. Saves bundle to `artifacts/transformer/<output_name>/`

Bundle contains: `model.pt` (weights), `tokenizer/` (saved tokenizer), `metrics.json`, `metadata.json`

---

### `inference/service.py` – Prediction Engine

`InferenceService` is the class that both the CLI and the API use to make predictions. It:

1. Reads `metadata.json` from the bundle to know what model type it is
2. If baseline: loads `estimator.joblib` (vectorizer + model)
3. If transformer: loads `model.pt` and the saved tokenizer
4. On `predict_text(text)`:
   - Runs preprocessing pipeline on the input
   - Vectorizes or tokenizes depending on model type
   - For SVM: uses decision function + sigmoid to get probability
   - For MuRIL: runs forward pass, applies softmax to logits
   - Returns label, confidence, probabilities, flagged tokens, normalized text

---

### `apps/api/main.py` – FastAPI Backend

Three endpoints:

**`GET /health`**
```json
{ "status": "ok", "bundle_loaded": true, "bundle_path": "artifacts/baseline/sample-baseline" }
```

**`GET /model-info`**
Returns model type, name, and its metrics.

**`POST /predict`**
Send:
```json
{ "text": "Tumhara attitude bohot cheap hai stop acting like clown" }
```
Get back:
```json
{
  "label": "Bullying",
  "confidence": 0.94,
  "probabilities": { "Non-Bullying": 0.06, "Bullying": 0.94 },
  "flagged_tokens": ["cheap", "clown"],
  "normalized_text": "tumhara attitude bohot [ABUSIVE] hai stop acting like [ABUSIVE]",
  "model_version": "linear_svm",
  "sub_category": "Harassment / Insult",
  "needs_review": false
}
```

Input validation is automatic — empty strings are rejected before they reach the model.

---

### `configs/` – YAML Config Files

Everything in Guardify is config-driven. You don't hardcode parameters — you change them in the YAML file.

**`configs/baseline.yaml`** — for SVM/LR:
```yaml
training:
  task: baseline
  test_size: 0.25
  val_size: 0.25
  output_name: sample-baseline
  kfold: 3

baseline:
  models: [logistic_regression, linear_svm]
  ngram_range: [1, 2]
  max_features: 3000
```

**`configs/muril.yaml`** — for MuRIL:
```yaml
training:
  task: transformer
  output_name: sample-muril

transformer:
  model_name: google/muril-base-cased
  epochs: 1
  batch_size: 16
  learning_rate: 0.00002
  max_length: 64
  dropout: 0.3
```

To add a real dataset, add it under `datasets:` in the YAML:
```yaml
datasets:
  - name: hasoc_train
    type: hasoc_csv
    path: data/external/hasoc_train.csv
    text_column: text
    label_column: task_1
```

---

## 🚀 Step-by-Step: Running the Project

### Step 1 – Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2 – Environment Check

```bash
PYTHONPATH=src venv/bin/python3 setup_test.py
```

Checks that PyTorch is working and detects your device (CPU / GPU / Apple MPS).

---

### Step 3 – Train Baseline Models (SVM + Logistic Regression)

```bash
PYTHONPATH=src venv/bin/python3 -m guardify.train --config configs/baseline.yaml
```

This runs `training/baseline.py`. The trained bundle is saved to:
```
artifacts/baseline/sample-baseline/
```

---

### Step 4 – Train MuRIL (Transformer)

```bash
PYTHONPATH=src venv/bin/python3 -m guardify.train --config configs/muril.yaml
```

First run downloads MuRIL (~500MB). Bundle saved to:
```
artifacts/transformer/sample-muril/
```

---

### Step 5 – Evaluate a Bundle

```bash
PYTHONPATH=src venv/bin/python3 -m guardify.evaluate \
  --config configs/baseline.yaml \
  --bundle artifacts/baseline/sample-baseline
```

Outputs accuracy, precision, recall, F1 and saves `evaluation_metrics.json` and `evaluation_misclassified.csv` inside the bundle folder.

---

### Step 6 – Predict from CLI

```bash
PYTHONPATH=src venv/bin/python3 -m guardify.predict \
  --bundle artifacts/baseline/sample-baseline \
  --text "Tumhara attitude bohot cheap hai"
```

---

### Step 7 – Run the API

```bash
GUARDIFY_MODEL_BUNDLE=artifacts/baseline/sample-baseline \
PYTHONPATH=src venv/bin/python3 -m uvicorn apps.api.main:app --reload
```

Visit `http://localhost:8000/docs` to test the API in your browser. This page is auto-generated by FastAPI.

---

### Step 8 – Run the Frontend

```bash
cd apps/web
npm install
npm run dev
```

Opens the React UI in your browser. By default, it calls `http://127.0.0.1:8000`; if your backend runs elsewhere, set `VITE_API_BASE_URL` before `npm run dev`.

---

### Step 9 – Add Real Data (When Ready)

1. Drop your CSV files into `data/external/`
2. Add them under `datasets:` in the relevant YAML config
3. Re-run training

The loader and schema automatically handle HASOC, TRAC, and custom formats.

---

## ⚠️ Common Issues & Solutions

**"Model bundle is not configured" from API**
You forgot to set the environment variable. Use:
```bash
GUARDIFY_MODEL_BUNDLE=artifacts/baseline/sample-baseline uvicorn apps.api.main:app
```

**"Config must declare at least one dataset"**
Your YAML file is missing a `datasets:` section or the path to the CSV is wrong.

**"Unknown label" error during training**
Your dataset has a label value the schema doesn't recognize. Add a `label_map` in the YAML config:
```yaml
datasets:
  - name: my_data
    type: custom_csv
    path: data/external/my_data.csv
    label_map:
      "1": "Bullying"
      "0": "Non-Bullying"
```

**Out of memory during MuRIL training**
Reduce `batch_size` in `configs/muril.yaml` to 1 or 2.

**Low accuracy / all zeros in metrics**
The demo `data/raw/data.csv` only has 2 rows — it is a fixture for testing, not real training data. You need to add real datasets (HASOC, TRAC, or your own) for meaningful results.

**"Module not found" errors**
Always run commands with `PYTHONPATH=src` prefix, or activate the venv first.

---

## 📊 Understanding Model Performance

After training, open `artifacts/<model_type>/<name>/metrics.json` to see results.

| Metric | What it means | Target |
|---|---|---|
| Accuracy | % of posts classified correctly | > 85% |
| F1 Macro | Balanced score across both classes | > 0.85 |
| Precision | Of posts predicted Bullying, how many actually were | > 0.85 |
| Recall | Of actual bullying posts, how many did we catch | > 0.85 |

Low recall = the model is missing real bullying (false negatives — dangerous).
Low precision = the model is flagging non-bullying as bullying (false positives — annoying).

Check `misclassified.csv` inside the bundle to see exactly what the model got wrong and improve from there.

---

## 📚 Glossary

| Term | Meaning |
|---|---|
| Epoch | One full pass through all training data |
| Batch | A group of samples processed together in one step |
| Loss | How wrong the model is — lower is better |
| F1 Score | Harmonic mean of precision and recall — best single metric for imbalanced data |
| Logits | Raw output numbers from a model before softmax |
| Softmax | Converts raw logits into probabilities that sum to 1 |
| Fine-tuning | Taking a pre-trained model (MuRIL) and continuing to train it on your specific task |
| Stratified split | Splitting data while preserving the same class ratio in each split |
| Bundle | A folder containing everything needed to load and run a trained model |
| TF-IDF | Numerical representation of text based on word frequency |
| AdamW | An optimizer — the algorithm that updates model weights during training |
| Attention mask | Tells the transformer which tokens are real content vs padding |
| Canonical format | The standardized column structure all datasets are converted to before training |