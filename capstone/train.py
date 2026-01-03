import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# =========================
# Text Cleaning
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
# Data Loading & Preprocessing
# =========================
def load_and_preprocess_data(data_path: str):
    columns = ["target", "id", "date", "flag", "user", "text"]

    df = pd.read_csv(
        data_path,
        header=None,
        encoding="latin-1",
        names=columns
    )

    # Keep only required columns
    df = df[["target", "text"]]

    # Map sentiment labels
    df["target"] = df["target"].map({0: 0, 4: 1})

    # Clean text
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    # Drop empty rows
    df = df[df["clean_text"] != ""].reset_index(drop=True)

    return df


# =========================
# Dataset Preparation
# =========================
def prepare_datasets(df: pd.DataFrame):
    X = df["clean_text"]
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    return X_train_vec, X_val_vec, y_train, y_val, vectorizer


# =========================
# Model Training & Evaluation
# =========================
def train_and_evaluate(X_train, X_val, y_train, y_val):
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("Final Model Performance")
    print("-" * 30)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("F1 Score:", f1_score(y_val, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred))

    return model


# =========================
# Model Saving
# =========================
def save_model(model, vectorizer, filename="sentiment_model.pkl"):
    with open(filename, "wb") as f_out:
        pickle.dump((model, vectorizer), f_out)

    print(f"Model and vectorizer saved to {filename}")


# =========================
# Main Entry Point
# =========================
def main():
    data_path = "./training.1600000.processed.noemoticon.csv"

    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)

    print("Preparing datasets...")
    X_train, X_val, y_train, y_val, vectorizer = prepare_datasets(df)

    print("Training model...")
    model = train_and_evaluate(X_train, X_val, y_train, y_val)

    print("Saving model...")
    save_model(model, vectorizer)


if __name__ == "__main__":
    main()
