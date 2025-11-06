import streamlit as st
import requests
import pickle
import os
import pandas as pd

model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

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

# --- When button clicked ---
if st.button("Predict Grade"):
    # Mappings
    mapping_gender = {"Male": 0, "Female": 1}
    mapping_ethnicity = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
    mapping_parentaledu = {"None": 0, "High School": 1, "Some College": 2, "Bachelor's": 3, "Higher": 4}
    mapping_support = {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}

    # Prepare data
    X = pd.DataFrame([{
        "age": age,
        "gender": mapping_gender[gender],
        "ethnicity": mapping_ethnicity[ethnicity],
        "parentaleducation": mapping_parentaledu[parentaleducation],
        "studytimeweekly": studytimeweekly,
        "absences": absences,
        "tutoring": 1 if tutoring == "Yes" else 0,
        "parentalsupport": mapping_support[parentalsupport],
        "extracurricular": 1 if extracurricular == "Yes" else 0,
        "sports": 1 if sports == "Yes" else 0,
        "music": 1 if music == "Yes" else 0,
        "volunteering": 1 if volunteering == "Yes" else 0
    }])

    try:
        prediction = model.predict(X)[0]
        grade_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
        st.success(f"🎯 Predicted Grade: **{grade_map[prediction]}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")