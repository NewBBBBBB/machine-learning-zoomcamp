import streamlit as st
import requests
import pickle
import os
import pandas as pd

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    # Handle both (model, dv) or just model
    if isinstance(obj, tuple):
        model, dv = obj
    else:
        model, dv = obj, None
    return model, dv

model, dv = load_model()

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓")

st.title("🎓 Student Grade Prediction App")
st.write("Enter the student's information below to predict their grade.")

# --- Input fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=15, max_value=18, value=17)
ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
parentaleducation = st.selectbox("Parental Education", ["None", "High School", "Some College", "Bachelor's", "Higher"])
studytimeweekly = st.number_input("Study Time Weekly (hours)", min_value=0, max_value=20, value=5)
absences = st.number_input("Absences", min_value=0, max_value=30, value=2)
tutoring = st.selectbox("Tutoring", ["Yes", "No"])
parentalsupport = st.selectbox("Parental Support", ["None", "Low", "Moderate", "High", "Very High"])
extracurricular = st.selectbox("Extracurricular", ["Yes", "No"])
sports = st.selectbox("Sports", ["Yes", "No"])
music = st.selectbox("Music", ["Yes", "No"])
volunteering = st.selectbox("Volunteering", ["Yes", "No"])

# --- Predict Button ---
if st.button("Predict Grade"):
    # Convert user input
    data = {
        "age": age,
        "gender": 0 if gender == "Male" else 1,
        "ethnicity": {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}[ethnicity],
        "parentaleducation": {"None": 0, "High School": 1, "Some College": 2, "Bachelor's": 3, "Higher": 4}[parentaleducation],
        "studytimeweekly": studytimeweekly,
        "absences": absences,
        "tutoring": 1 if tutoring == "Yes" else 0,
        "parentalsupport": {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}[parentalsupport],
        "extracurricular": 1 if extracurricular == "Yes" else 0,
        "sports": 1 if sports == "Yes" else 0,
        "music": 1 if music == "Yes" else 0,
        "volunteering": 1 if volunteering == "Yes" else 0
    }

    # Transform if DictVectorizer exists
    if dv is not None:
        X = dv.transform([data])
    else:
        X = pd.DataFrame([data])

    # Predict
    pred = model.predict(X)[0]
    grade_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
    st.success(f"🎯 Predicted Grade: **{grade_map[pred]}**")