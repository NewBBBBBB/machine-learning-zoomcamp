# 🎓 Students Performance Prediction using Machine Learning  
### Machine Learning Zoomcamp Midterm Project  

---

## 📘 Dataset Description  

This project uses the **[Students Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)** created by *Rabie El Kharoua (2023)*.  
The dataset contains **2,392 high school student records**, covering demographic, academic, and behavioral information.  
The goal is to build a **machine learning model** that predicts a student’s final **GradeClass** (A–F) based on their study habits, parental involvement, and extracurricular activities.  

---

## 🧩 Attribute Information  

### 👤 Student Information
| Feature | Description |
|----------|--------------|
| **StudentID** | Unique identifier for each student (1001–3392) |

### 🧠 Demographic Details
| Feature | Description |
|----------|--------------|
| **Age** | Student’s age (15–18 years) |
| **Gender** | 0 = Male, 1 = Female |
| **Ethnicity** | 0 = Caucasian, 1 = African American, 2 = Asian, 3 = Other |
| **ParentalEducation** | 0 = None, 1 = High School, 2 = Some College, 3 = Bachelor’s, 4 = Higher |

### 📚 Study Habits
| Feature | Description |
|----------|--------------|
| **StudyTimeWeekly** | Weekly study time (hours, 0–20) |
| **Absences** | Number of absences per year (0–30) |
| **Tutoring** | 0 = No, 1 = Yes |

### 👨‍👩‍👧 Parental Involvement
| Feature | Description |
|----------|--------------|
| **ParentalSupport** | Level of support (0 = None, 1 = Low, 2 = Moderate, 3 = High, 4 = Very High) |

### 🎭 Extracurricular Activities
| Feature | Description |
|----------|--------------|
| **Extracurricular** | 0 = No, 1 = Yes |
| **Sports** | 0 = No, 1 = Yes |
| **Music** | 0 = No, 1 = Yes |
| **Volunteering** | 0 = No, 1 = Yes |

### 🎓 Academic Performance
| Feature | Description |
|----------|--------------|
| **GPA** | Grade Point Average (2.0–4.0 scale) — directly influences grade classification |

### 🎯 Target Variable
| Feature | Description |
|----------|--------------|
| **GradeClass** | Classification of grades based on GPA:<br>0 = A (≥3.5), 1 = B (3.0–3.49), 2 = C (2.5–2.99), 3 = D (2.0–2.49), 4 = F (<2.0) |

---

## 🎯 Problem Statement  

Academic success is influenced by multiple factors — study behavior, parental involvement, and extracurricular engagement.  
The aim of this project is to **predict a student's academic performance (GradeClass)** using these behavioral and demographic variables.  

This project seeks to:  
- Identify the most significant predictors of student performance  
- Build classification models to predict grade outcomes  
- Analyze fairness and bias in predictions (especially across demographic groups)  

---

## 🚀 Tools & Libraries  

- **Python 3.10+**  
- **Pandas**, **NumPy** — data manipulation and preprocessing  
- **Seaborn**, **Matplotlib** — visualization and correlation analysis  
- **Scikit-learn** — model building, evaluation, and hyperparameter tuning    
- **Jupyter Notebook** — experimentation and reporting  

---

## 🧠 Machine Learning Approach  

1. **Data Preprocessing**
   - Removed redundant features (e.g., GPA to avoid data leakage)
   - Handled missing values (if any)
   - Performed **categorical mapping and encoding** (e.g., converting 0/1 to “No/Yes” or vice versa)
   - Ensured consistent column naming (lowercase and underscore formatting) 

2. **Exploratory Data Analysis (EDA)**
   - Correlation & Mutual Information analysis  
   - Feature importance with Random Forest  

3. **Model Development**
   - Logistic Regression  
   - Random Forest Classifier  
   - Decision Tree (baseline only)

4. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score (Macro)  
   - Cross-validation with `GridSearchCV` for tuning  

5. **Final Comparison**
   - Compare baseline vs tuned models in one results table  

---

## 🧩 Acknowledgment  

Dataset by **Rabie El Kharoua** (2023) —  
📊 *Students Performance Dataset: Academic Success Factors in High School Students*  
Licensed under **CC BY 4.0**.  
Link: [https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)

---

## 🏁 Summary  

This project demonstrates how machine learning can be applied to **predict academic success** based on real-world student behavior and family support data.  
The resulting model aims to assist educators in identifying students at risk and implementing early interventions.
