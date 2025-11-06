
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Data Loading & Preprocessing
def load_and_preprocess_data(data_path: str):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()

    # Drop unnecessary columns
    df = df.drop(columns=['studentid', 'gpa'], errors='ignore')

    # Define mapping dictionaries
    gender_map = {0: 'Male', 1: 'Female'}
    ethnicity_map = {0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'}
    parental_edu_map = {0: 'None', 1: 'High School', 2: 'Some College', 3: "Bachelor's", 4: 'Higher'}
    parental_support_map = {0: 'None', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Very High'}
    grade_class_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

    yes_no_map = {0: 'No', 1: 'Yes'}

    # Apply mappings
    df['gender'] = df['gender'].map(gender_map)
    df['ethnicity'] = df['ethnicity'].map(ethnicity_map)
    df['parentaleducation'] = df['parentaleducation'].map(parental_edu_map)
    df['parentalsupport'] = df['parentalsupport'].map(parental_support_map)
    df['gradeclass'] = df['gradeclass'].map(grade_class_map)

    for col in ['tutoring', 'extracurricular', 'sports', 'music', 'volunteering']:
        df[col] = df[col].map(yes_no_map)

    return df


# Data Splitting & Encoding
def prepare_datasets(df: pd.DataFrame):
    X = df.drop(columns=["gradeclass"])
    y = df["gradeclass"]

    # Split data: train/val/test (60/20/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # DictVectorizer for encoding
    dv = DictVectorizer(sparse=False)
    X_train_encoded = dv.fit_transform(X_train.to_dict(orient="records"))
    X_val_encoded = dv.transform(X_val.to_dict(orient="records"))

    return X_train_encoded, X_val_encoded, y_train, y_val, dv


# Model Training & Evaluation
def train_and_evaluate(X_train, X_val, y_train, y_val):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, average='macro'),
        "Recall": recall_score(y_val, y_pred, average='macro'),
        "F1 Score": f1_score(y_val, y_pred, average='macro')
    }

    print("Final Model Performance")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return model, metrics


# Model Saving 
def save_model(model, dv, filename='logistic_regression_model.pkl'):
    with open(filename, 'wb') as f_out:
        pickle.dump((model, dv), f_out)
    print(f"Model and DictVectorizer saved to {filename}")


#  Main Entry Point 
def main():
    data_path = '/workspaces/machine-learning-zoomcamp/07-midterm/Student_performance_data.csv'
    df = load_and_preprocess_data(data_path)
    X_train, X_val, y_train, y_val, dv = prepare_datasets(df)
    model, metrics = train_and_evaluate(X_train, X_val, y_train, y_val)
    save_model(model, dv)


if __name__ == "__main__":
    main()
