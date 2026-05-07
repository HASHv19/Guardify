# 🛡️ Guardify

**Guardify** is a bilingual cyberbullying detection platform engineered to identify abusive content in **English, Hinglish, and transliterated Hindi** social media text. 

Built as a complete end-to-end NLP pipeline, Guardify features deterministic preprocessing, modular dataset adapters, and a production-ready inference service. It includes both high-performance ML baselines (SVM/Logistic Regression) and advanced Transformer support (Google MuRIL), served via a FastAPI backend and a responsive React frontend.

## 🌐 Live Demo

The project is actively deployed on free-tier cloud infrastructure:
- **Frontend (Vercel):** [https://guardify-taupe.vercel.app](https://guardify-taupe.vercel.app)
- **Backend API (Render):** [https://guardify-tq9e.onrender.com](https://guardify-tq9e.onrender.com)

*(Note: The Render backend sleeps after 15 minutes of inactivity. The first request may take ~50 seconds to wake up the server.)*

## ✨ Key Features

- **Bilingual NLP Pipeline**: Tailored preprocessing for Indian social media text (URL cleanup, emoji mapping, obfuscation resolution, and abuse lexicon flagging).
- **Extensible Datasets**: Canonical data ingestion for standard sources (HASOC, TRAC) via YAML configurations.
- **Dual Model Architecture**: 
  - Extremely fast baseline classifiers (TF-IDF + Linear SVM/LR) perfect for edge/CPU inference.
  - Transformer training flows (MuRIL) for deep contextual understanding.
- **Production API**: Robust FastAPI implementation with pre-loaded model bundles for sub-millisecond inference.
- **Interactive UI**: Modern React frontend for real-time moderation demonstration.
- **Cloud-Ready**: Fully Dockerized and configured for PaaS deployments (Render, Vercel, Heroku).

## 📁 Project layout

```text
Guardify/
├── apps/
│   ├── api/              # FastAPI backend
│   └── web/              # React frontend
├── configs/              # Training configs
├── data/
│   ├── raw/              # Local fixture data
│   └── external/         # Drop larger datasets here
├── src/guardify/         # Core NLP package
├── tests/                # Python test suite
├── example_workflow.py   # Guided workflow script
└── setup_test.py         # Environment smoke check
```

## 🚀 Quick Start (Local Demo)

The repository includes a lightweight, pre-trained baseline model (`sample-baseline`) so you can run the API and frontend immediately without training.

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Backend API

```bash
GUARDIFY_MODEL_BUNDLE=artifacts/baseline/sample-baseline \
PYTHONPATH=src venv/bin/python3 -m uvicorn apps.api.main:app --reload
```
*API will be available at `http://localhost:8000/docs`.*

### 3. Start the Frontend UI

Open a new terminal:
```bash
cd apps/web
npm install
npm run dev
```
*UI will be available at `http://localhost:5173` (default Vite port).*

### 4. CLI Inference

You can also test the model directly from the command line:
```bash
PYTHONPATH=src venv/bin/python3 -m guardify.predict \
  --bundle artifacts/baseline/sample-baseline \
  --text "Tumhara attitude bohot cheap hai"
```

## 🧠 Training & Transformers

Guardify is entirely config-driven. You can train your own models by providing datasets in `data/external/` and running the training scripts.

```bash
# Train a new baseline bundle
PYTHONPATH=src venv/bin/python3 -m guardify.train --config configs/baseline.yaml
```

**Note on Transformers:** Guardify natively supports fine-tuning transformer architectures like Google MuRIL (`configs/muril.yaml`). To keep this repository lightweight and fast to clone, heavy transformer weights (~900MB) are **not tracked** in version control. You can generate them locally anytime using the provided training pipeline.

## 📡 API Contract

### `POST /predict`

**Request:**
```json
{ "text": "Tumhara attitude bohot cheap hai stop acting like clown" }
```

**Response:**
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
