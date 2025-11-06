import streamlit as st
import requests

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
    student_data = [{
        "gender": gender,
        "age": age,
        "ethnicity": ethnicity,
        "parentaleducation": parentaleducation,
        "studytimeweekly": studytimeweekly,
        "absences": absences,
        "tutoring": tutoring,
        "parentalsupport": parentalsupport,
        "extracurricular": extracurricular,
        "sports": sports,
        "music": music,
        "volunteering": volunteering
    }]

    # URL of your Flask API
    url = "http://localhost:9696/predict"

    try:
        response = requests.post(url, json=student_data)
        if response.status_code == 200:
            result = response.json()[0]["predicted_grade"]
            st.success(f"🎯 Predicted Grade: **{result}**")
        else:
            st.error(f"API error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
