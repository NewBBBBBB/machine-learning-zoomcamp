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
```

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

## ⚙️ Deployment Methods  

### 🔹 1️⃣ Run Locally

This method runs the project on your local machine using **Pipenv** for dependency management.

---

#### Prerequisites
- Python 3.10 or newer  
- Pipenv (`pip install pipenv`)  
- Sentiment140 dataset downloaded from Kaggle  

---

#### Steps

##### 1) Clone the repository
```bash
git clone https://github.com/NewBBBBBB/machine-learning-zoomcamp.git
cd machine-learning-zoomcamp/capstone
```
##### 2) Download the dataset
Download the Sentiment140 dataset from Kaggle: https://www.kaggle.com/datasets/kazanova/sentiment140
Place the CSV file directly inside the `capstone/` directory (same location as `train.py`):


##### 3) Install project dependencies
```bash
pipenv install
```

##### 4) Activate the virtual environment
```bash
pipenv shell
```

##### 5) Train the sentiment analysis model
```bash
python train.py
```

##### 6) Run Streamlit application (UI)
```bash
streamlit run app.py
```
Once started, open your browser at:
```text
http://localhost:8501
```

##### 7) Run FastAPI application (API)
Run FastAPI Locally
```bash
uvicorn app:app --reload
```

The API will be available at:
```text
http://127.0.0.1:8000
```

Interactive API documentation can be accessed at:
```text
http://127.0.0.1:8000/docs
```

### 🔹 2️⃣ Run via Docker
> Note: Docker deployment is intended for **inference only**.  
> Model training is performed locally using Pipenv.
```bash
docker build -t twitter-sentiment-api .
# Run container
docker run -p 9696:9696 twitter-sentiment-api
```
Once the container is running, the API will be available at:
```text
http://localhost:9696
```
### API Endpoint

**POST** `/predict`

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


## 🧩 Acknowledgment  

Dataset by **Kazanova** (2009) —  
📊 *Sentiment140 Dataset with 1.6 million tweets*  
Licensed under **CC BY 4.0**.  
Link: [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

---