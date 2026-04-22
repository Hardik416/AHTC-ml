# AHTC-ml

AI vs Human Text Detector built with FastAPI and scikit-learn.

## Overview

This project trains a text-classification ensemble model and serves predictions through a FastAPI backend with a simple web UI.

- Training script: `train.py`
- API server: `main.py`
- Frontend page: `index.html`
- Saved model output: `models/detector_model.pkl`

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

Run:

```bash
python train.py
```

This downloads the DAIGT dataset via `kagglehub`, trains a soft-voting ensemble (Logistic Regression + Naive Bayes + SVM), and saves:

`models/detector_model.pkl`

## Run the API

Start the server:

```bash
uvicorn main:app --reload
```

Open in browser:

`http://127.0.0.1:8000/`

## API endpoint

### `POST /predict`

Request body:

```json
{
  "text": "Your input text here"
}
```

Response example:

```json
{
  "verdict": "AI Generated",
  "confidence": "92.31%"
}
```

## Notes

- `main.py` expects `models/detector_model.pkl` to exist before starting.
- The frontend currently calls `http://127.0.0.1:8000/predict` directly.
