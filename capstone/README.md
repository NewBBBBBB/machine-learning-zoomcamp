# 🎭 Twitter Sentiment Analysis using Machine Learning  
### Machine Learning Zoomcamp Midterm Project  

🚀 **Live Demo:** *(Optional – add link if deployed)*  
---  

## 📘 Dataset Description  

This project uses the **Sentiment140 Dataset** created by *Kazanova (2009)*.  
The dataset contains **1,600,000 labeled tweets**, collected via the Twitter API, with sentiment polarity labels.

The objective is to build a **machine learning model** that predicts whether a tweet expresses **positive or negative sentiment** based solely on its text content.

---

## 🧩 Attribute Information  

### 📱 Tweet Information  

| Feature | Description |
|--------|-------------|
| **target** | Sentiment label: 0 = Negative, 4 = Positive |
| **id** | Unique identifier for each tweet |
| **date** | Timestamp when the tweet was posted |
| **flag** | Query flag (typically `NO_QUERY`) |
| **user** | Twitter username |
| **text** | Raw tweet content |

> Only the `target` and `text` columns are used in this project.  
> All other metadata fields were removed as they do not contribute meaningful signal for sentiment prediction.

---

## 🎯 Problem Statement  

Social media platforms generate large volumes of unstructured text reflecting opinions, emotions, and reactions.  
The goal of this project is to **automatically classify tweet sentiment** as **positive or negative** using classical NLP and machine learning techniques.

This project aims to:
- Understand the structure and characteristics of real-world tweet data  
- Extract meaningful textual features for sentiment prediction  
- Compare multiple linear classification models  
- Build a reproducible and deployable sentiment analysis pipeline  

---

## 📥 Dataset Download Instructions  

The Sentiment140 dataset is **not included** in this repository due to its large file size.

Please download the dataset manually from Kaggle:

https://www.kaggle.com/datasets/kazanova/sentiment140

After downloading, place the CSV file **directly in the project root directory**, i.e. the same folder where `train.py` is located:

```text
training.1600000.processed.noemoticon.csv
---

## 🧠 Machine Learning Approach  

### A. Data Preparation & Cleaning
- Removed non-text metadata columns
- Converted sentiment labels (0 → negative, 4 → positive)
- Normalized text by:
  - Lowercasing
  - Removing URLs and user mentions
  - Removing punctuation, numbers, and special characters
  - Normalizing whitespace
- No stemming or lemmatization was applied to preserve feature interpretability

### B. Exploratory Data Analysis (EDA)
- Analyzed tweet length and word count distributions
- Verified class balance between positive and negative tweets
- Inspected common words and sentiment-bearing terms
- Examined noise patterns such as URLs, mentions, and hashtags

### C. Feature Engineering
- Used **TF-IDF vectorization**
- Evaluated both:
  - Unigrams `(1,1)`
  - Unigrams + bigrams `(1,2)`
- Stopwords were removed during vectorization
- Bigram features helped capture sentiment phrases (e.g. *“not good”*)

### D. Feature Importance Analysis
Two complementary methods were used:
- **Logistic Regression coefficients** (model-based interpretability)
- **Chi-square statistics** (model-agnostic validation)

Both methods consistently highlighted emotionally meaningful words such as *“happy”*, *“love”*, *“sad”*, and *“disappointed”*.

### E. Model Selection & Tuning
Models evaluated:
- Logistic Regression (baseline and tuned)
- Linear Support Vector Machine (baseline and tuned)

Hyperparameter tuning was performed using **GridSearchCV** with 3-fold cross-validation.

The tuned **Logistic Regression model** was selected as the final model due to:
- Competitive performance
- Strong interpretability
- Availability of probability estimates (important for deployment)

---

## 📊 Model Comparison Summary  

| Model | Accuracy | F1 Score |
|-----|----------|----------|
| Logistic Regression (Baseline) | ~0.76 | ~0.77 |
| Linear SVM (Baseline) | ~0.77 | ~0.77 |
| Logistic Regression (Tuned) | ~0.79 | ~0.80 |
| Linear SVM (Tuned) | ~0.78 | ~0.79 |

---

## ⚙️ Deployment  

The final trained model is deployed using two approaches:

1. **Streamlit Web Application** – for interactive user-facing predictions  
2. **FastAPI Application (Local)** – for API-based inference and integration  

---

## 🚀 Streamlit Deployment
A Streamlit application is used to provide a simple and interactive interface where users can input tweet text and receive sentiment predictions in real time.

### Features
- Text input for tweet content  
- Displays predicted sentiment (Positive / Negative)  
- Shows model confidence score  
- Lightweight and user-friendly UI  

### Run Streamlit Locally
```bash
streamlit run app.py
```

Once started, the application will be available at:
```bash
http://localhost:8501
```
This deployment is suitable for demonstrations and exploratory use.

---

## 🌐 FastAPI Deployment (Local)

In addition to Streamlit, the model is exposed via a **FastAPI** application to support programmatic access.

### API Endpoint

**POST** `/predict_sentiment`

### Request Body
```json
{
  "text": "I really love this product!"
}
```
Response

```json
{
  "sentiment": "positive",
  "probability": 0.87
}
```
Run FastAPI Locally
```bash
uvicorn app:app --reload
```

The API will be available at:
```bash
http://127.0.0.1:8000
```

Interactive API documentation can be accessed at:
```bash
http://127.0.0.1:8000/docs
```

## 🧩 Acknowledgment  

Dataset by **Kazanova** (2009) —  
📊 *Sentiment140 Dataset with 1.6 million tweets*  
Licensed under **CC BY 4.0**.  
Link: [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

---