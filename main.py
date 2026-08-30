from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Churn Prediction API")

with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)


class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    gender_Male: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    input_dict = customer.dict()
    
    input_dict['InternetService_Fiber optic'] = input_dict.pop('InternetService_Fiber_optic')
    input_dict['Contract_One year'] = input_dict.pop('Contract_One_year')
    input_dict['Contract_Two year'] = input_dict.pop('Contract_Two_year')
    input_dict['PaymentMethod_Credit card (automatic)'] = input_dict.pop('PaymentMethod_Credit_card')
    input_dict['PaymentMethod_Electronic check'] = input_dict.pop('PaymentMethod_Electronic_check')
    input_dict['PaymentMethod_Mailed check'] = input_dict.pop('PaymentMethod_Mailed_check')
    
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]
    
    numeric_cols = ['tenure', 'MonthlyCharges']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    probability = model.predict_proba(input_df)[0][1]
    prediction = "Yes" if probability >= 0.70 else "No"
    
    return {
        "churn_probability": round(float(probability), 3),
        "churn_prediction": prediction,
        "threshold_used": 0.70
    }



