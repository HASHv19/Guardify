# Guardify

Guardify is a bilingual cyberbullying detection platform for English and Hinglish social text. This version upgrades the original starter scripts into a modular NLP package with dataset adapters, deterministic preprocessing, baseline and transformer training flows, a FastAPI backend, and a React frontend.

## 🌐 Live Demo

The project is actively deployed on free-tier cloud infrastructure:
- **Frontend (Vercel):** [https://guardify-taupe.vercel.app](https://guardify-taupe.vercel.app)
- **Backend API (Render):** [https://guardify-tq9e.onrender.com](https://guardify-tq9e.onrender.com)

*(Note: The Render backend sleeps after 15 minutes of inactivity. The first request may take ~50 seconds to wake up the server.)*

## What is included

- Canonical dataset ingestion for sample, HASOC-style, TRAC-style, CSV, and JSON sources
- Ordered preprocessing with URL cleanup, emoji mapping, obfuscation cleanup, repeated-character normalization, and flagged-token extraction
- Baseline classifiers with TF-IDF + Logistic Regression / Linear SVM
- Transformer training path using MuRIL as the default multilingual model
- Shared model bundle format for training, evaluation, and API inference
- FastAPI endpoints for health, model info, and prediction
- React app for quick interactive moderation demos
- **Dockerized Backend** ready for PaaS deployment (Render, Heroku, etc.)

## Project layout

```text
Guardify/
├── apps/
│   ├── api/              # FastAPI backend
│   └── web/              # React frontend
├── configs/              # Training configs
├── data/
│   ├── raw/              # Local fixture data
│   └── external/         # Drop larger datasets here
├── src/guardify/         # Core package
├── tests/                # Python test suite
├── example_workflow.py   # Guided workflow script
└── setup_test.py         # Environment smoke check
```

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

If you prefer a package-style install, `requirements.txt` installs the project in editable mode.

## Quick start

### 1. Run the environment smoke test

```bash
PYTHONPATH=src venv/bin/python3 setup_test.py
```

### 2. Train a baseline bundle

```bash
PYTHONPATH=src venv/bin/python3 -m guardify.train --config configs/baseline.yaml
```

### 3. Evaluate a bundle

```bash
PYTHONPATH=src venv/bin/python3 -m guardify.evaluate \
  --config configs/baseline.yaml \
  --bundle artifacts/baseline/sample-baseline
```

### 4. Run the API

```bash
GUARDIFY_MODEL_BUNDLE=artifacts/baseline/sample-baseline \
PYTHONPATH=src venv/bin/python3 -m uvicorn apps.api.main:app --reload
```

### 5. Predict from the CLI

```bash
PYTHONPATH=src venv/bin/python3 -m guardify.predict \
  --bundle artifacts/baseline/sample-baseline \
  --text "Tumhara attitude bohot cheap hai"
```

### 6. Start the frontend

```bash
cd apps/web
npm install
npm run dev
```

By default, the frontend calls `http://127.0.0.1:8000`. If your backend runs on a different URL or port, set `VITE_API_BASE_URL` before starting Vite.

## API contract

### `POST /predict`

Request:

```json
{ "text": "Tumhara attitude bohot cheap hai stop acting like clown" }
```

Response:

```json
{
  "label": "Bullying",
  "confidence": 0.94,
  "probabilities": {
    "Non-Bullying": 0.06,
    "Bullying": 0.94
  },
  "flagged_tokens": ["cheap", "clown"],
  "normalized_text": "tumhara attitude bohot [ABUSIVE] hai stop acting like [ABUSIVE]",
  "model_version": "linear_svm",
  "sub_category": "Harassment / Insult",
  "needs_review": false
}
```

## Config-driven datasets

Each dataset adapter must be normalized into:

- `id`
- `text`
- `original_label`
- `binary_label`
- `source`
- `language_hint`
- `split`

Example dataset config:

```yaml
datasets:
  - name: hasoc_train
    type: hasoc_csv
    path: data/external/hasoc_train.csv
    text_column: text
    label_column: task_1
```

## Notes

- The bundled `data/raw/data.csv` file is only a demo fixture.
- MuRIL is the default transformer path; XLM-R, Bi-LSTM, and hybrid models are future extensions.
- `flagged_tokens` are lexicon/rule-based in this version, not model explanations.

## Deployment

The project is configured for easy deployment to containerized PaaS platforms (like Render or Heroku) and static frontend hosts (like Vercel or Netlify).

### Backend (Docker / Render)
1. The backend uses the provided `Dockerfile`.
2. Ensure you set `GUARDIFY_MODEL_BUNDLE` and `CORS_ORIGINS` in the deployment environment.
3. The server automatically binds to `$PORT` making it compatible with cloud providers.

### Frontend (Vite / Vercel)
1. Point your host to the `apps/web` root directory.
2. Set the `VITE_API_BASE_URL` environment variable to your live backend URL (e.g., `https://my-backend.onrender.com`).
3. Deploy as a standard Vite React app.

## Documentation

Beginner-oriented notes are still available in [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md).
