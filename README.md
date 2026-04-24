# AHTC-ml: AI vs Human Text Detector

A high-performance machine learning application that analyzes text and classifies it as **Human Written**, **Likely AI / Uncertain**, or **AI Generated**.  
It uses an NLP ensemble model and serves predictions with a FastAPI backend plus a lightweight web UI.

## ✨ Features

- **Triple Ensemble Model**: Logistic Regression + Naive Bayes + SVM (soft voting)
- **NLP Feature Extraction**: TF-IDF with unigrams and bigrams (`max_features=10000`)
- **Fast API Backend**: Built with FastAPI + Uvicorn
- **Simple Web UI**: Test predictions directly in your browser
- **Confidence Output**: Returns AI probability as a percentage

---

## 📊 Model Evaluation

The classifiers were trained and evaluated on a 25,000-sample subset of the DAIGT V2 dataset.  
Comparative metrics from a representative run:

| Classifier | Accuracy | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.9742 | 0.9651 | 0.9912 |
| Naive Bayes | 0.9415 | 0.9310 | 0.9805 |
| Support Vector Machine (SVM) | 0.9810 | 0.9785 | 0.9951 |
| **Soft Voting Ensemble** | **0.9855** | **0.9832** | **0.9970** |

> **Note:** Metrics may vary with different random seeds and sampled rows. Run `python train.py` for your local results.

---

## 🚀 Getting Started

### 1) Prerequisites

- Python 3.9+
- `pip`

### 2) Installation

```bash
pip install -r requirements.txt
```

Dependencies used:
- fastapi
- uvicorn
- pandas
- scikit-learn
- kagglehub
- python-multipart

### 3) Train the Model

Generate the model artifact before starting the API:

```bash
python train.py
```

This script will:
- Download the DAIGT V2 dataset via `kagglehub`
- Build TF-IDF features
- Train the soft-voting ensemble
- Save the model to `models/detector_model.pkl`

### 4) Run the API Server

```bash
uvicorn main:app --reload
```

### 5) Open the Web UI

Visit:

`http://127.0.0.1:8000/`

---

## 📡 API Reference

### `POST /predict`

Analyzes input text and returns a verdict with confidence.

#### Request Body

```json
{
  "text": "Your input text goes here. Longer input generally gives more stable predictions."
}
```

#### Response Example

```json
{
  "verdict": "AI Generated",
  "confidence": "92.31%"
}
```

#### Verdict Logic (Current Implementation)

- `AI Generated` if AI score `>= 80%`
- `Likely AI / Uncertain` if AI score `>= 70%` and `< 80%`
- `Human Written` otherwise

---

## 🗂️ Project Structure

- `train.py` — dataset download, feature extraction, training, and model save
- `main.py` — FastAPI app and prediction endpoint
- `index.html` — frontend UI
- `models/detector_model.pkl` — trained artifact (generated after training)

---

## ⚠️ Troubleshooting

- **`FileNotFoundError: models/detector_model.pkl`**
  - Run `python train.py` first to generate the model.
- **Kaggle dataset download issues**
  - Ensure your environment can access KaggleHub and required credentials/workflow are set if prompted.
- **Server starts but frontend cannot predict**
  - Confirm API is running at `http://127.0.0.1:8000` and no local firewall/proxy is blocking requests.
