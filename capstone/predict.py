import pickle
import re
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# Load model & vectorizer
# =========================
MODEL_PATH = "sentiment_model.pkl"

with open(MODEL_PATH, "rb") as f_in:
    model, vectorizer = pickle.load(f_in)

# =========================
# Text Cleaning (MUST match train.py)
# =========================
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
MULTI_SPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# =========================
# FastAPI App
# =========================
app = FastAPI(title="Tweet Sentiment Analysis API")


# =========================
# Request / Response Schemas
# =========================
class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    sentiment: str
    probability: float


# =========================
# Health Check
# =========================
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sentiment API is running"}


# =========================
# Prediction Endpoint
# =========================
@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: TextRequest):
    cleaned_text = clean_text(request.text)

    X = vectorizer.transform([cleaned_text])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    sentiment = "positive" if prediction == 1 else "negative"
    confidence = float(max(proba))

    return {
        "sentiment": sentiment,
        "probability": confidence
    }
