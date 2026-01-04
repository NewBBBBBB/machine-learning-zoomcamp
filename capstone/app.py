import streamlit as st
import pickle
import re
from pathlib import Path

# =========================
# Load model & vectorizer
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
MODEL_PATH = "sentiment_model.pkl"

st.write("Loading model from:", MODEL_PATH)
with open(MODEL_PATH, "rb") as f_in:
    model, vectorizer = pickle.load(f_in)

# =========================
# Text Cleaning (same as train.py & predict.py)
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
# Streamlit UI
# =========================
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")

st.title("🎭 Twitter Sentiment Analysis")
st.write("Enter a tweet below to predict its sentiment.")

user_input = st.text_area("Tweet text", height=120)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        X = vectorizer.transform([cleaned])

        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = max(proba)

        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

        st.subheader("Prediction Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2%}")