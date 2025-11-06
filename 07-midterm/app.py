import pickle
import pandas as pd
import streamlit as st
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("🎓 Student Performance Predictor")

# Input fields
age = st.number_input("Age", 15, 18, 16)
study_time = st.number_input("Study Time Weekly (hours)", 0, 20, 10)
absences = st.number_input("Absences (per year)", 0, 30, 3)
parent_support = st.slider("Parental Support (0-4)", 0, 4, 2)
tutoring = st.selectbox("Tutoring", ["No", "Yes"])
extracurricular = st.selectbox("Extracurricular", ["No", "Yes"])

# Convert to numeric
tutoring = 1 if tutoring == "Yes" else 0
extracurricular = 1 if extracurricular == "Yes" else 0

# Predict
if st.button("Predict Grade Class"):
    X = pd.DataFrame([{
        "age": age,
        "studytimeweekly": study_time,
        "absences": absences,
        "parentalsupport": parent_support,
        "tutoring": tutoring,
        "extracurricular": extracurricular
    }])
    prediction = model.predict(X)[0]
    grade_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
    st.success(f"Predicted Grade: {grade_map[prediction]}")
