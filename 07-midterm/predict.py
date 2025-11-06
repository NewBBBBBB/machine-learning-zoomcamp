
import pickle
from flask import Flask, request, jsonify

with open('logistic_regression_model.pkl', 'rb') as f_in:
    model, dv = pickle.load(f_in)

# Initialize Flask app
app = Flask('student-grade-prediction')

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON array of student records, e.g.:
    [
        {
            "gender": "Male",
            "age": 18,
            "ethnicity": "Asian",
            "parentaleducation": "Bachelor's",
            "studytimeweekly": 8,
            "absences": 3,
            "tutoring": "No",
            "parentalsupport": "Moderate",
            "extracurricular": "Yes",
            "sports": "No",
            "music": "Yes",
            "volunteering": "Yes"
        }
    ]
    """
    student_data = request.get_json()

    # Convert JSON input into vectorized form
    X = dv.transform(student_data)

    # Predict grade class (A, B, C, D, F)
    preds = model.predict(X)

    # Return predictions as JSON
    results = [{"predicted_grade": pred} for pred in preds]
    return jsonify(results)


# Root endpoint for health check
@app.route('/')
def index():
    return jsonify({"message": "Student Grade Prediction API is running!"})


# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)
