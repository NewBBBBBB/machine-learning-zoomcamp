import pickle
from fastapi import FastAPI
from pydantic import BaseModel

model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: float
    annual_income: float

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Lead scoring API is running"}

@app.post("/predict")
def predict(lead: Lead):
    lead_dict = lead.dict()
    prob = model.predict_proba([lead_dict])[0, 1]
    return {"probability": round(float(prob), 3)}
