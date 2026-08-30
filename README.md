# Churn Prediction API

A production-style REST API wrapping the [Telco Customer Churn model](https://github.com/YOUR-USERNAME/churn-prediction) in a FastAPI service, so churn predictions can be integrated directly into other systems (a CRM, a scheduled job, a client's own app) rather than only accessed through a UI.

This complements the [Streamlit demo](https://churn-prediction-app-8accxu39venyj7muejywtc.streamlit.app/) built for the same model — the Streamlit app is for a human to interactively explore predictions; this API is for a developer to integrate predictions into their own software.

## What It Does

- Loads the trained logistic regression model, fitted scaler, and expected feature columns (same artifacts used in the Streamlit app)
- Exposes a `POST /predict` endpoint: send a customer's data, get back a churn probability and a Yes/No prediction at the same 0.70 business-driven threshold established in the original churn analysis
- Automatic request validation via Pydantic — malformed or incomplete requests are rejected with clear, structured error messages before ever reaching the model
- Interactive, auto-generated API documentation at `/docs` (Swagger UI) — no separate documentation effort required

## Example Request

```json
POST /predict

{
  "tenure": 1,
  "MonthlyCharges": 95.0,
  "SeniorCitizen": 0,
  "Partner": 0,
  "Dependents": 0,
  "PhoneService": 1,
  "MultipleLines": 0,
  "OnlineSecurity": 0,
  "OnlineBackup": 0,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 1,
  "PaperlessBilling": 1,
  "gender_Male": 1,
  "InternetService_Fiber_optic": 1,
  "InternetService_No": 0,
  "Contract_One_year": 0,
  "Contract_Two_year": 0,
  "PaymentMethod_Credit_card": 0,
  "PaymentMethod_Electronic_check": 1,
  "PaymentMethod_Mailed_check": 0
}
```

## Example Response

```json
{
  "churn_probability": 0.915,
  "churn_prediction": "Yes",
  "threshold_used": 0.7
}
```

## Running Locally

```bash
git clone <this-repo-url>
cd churn-prediction-api
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive documentation, or send requests directly to `http://127.0.0.1:8000/predict`.

## Repository Structure

```
├── README.md
├── main.py                # FastAPI app: model loading, request validation, prediction endpoint
├── requirements.txt
├── churn_model.pkl         # Trained logistic regression model
├── scaler.pkl              # Fitted StandardScaler for tenure/MonthlyCharges
└── model_columns.pkl       # Expected feature columns/order for the model
```

## Tools Used

Python, FastAPI, Uvicorn, Pydantic, scikit-learn, pandas

## Notes

- The 0.70 decision threshold reflects a deliberate business tradeoff (balancing false alarms against missed churners) made and justified during the original model development — see the [churn analysis repo](https://github.com/AmiraScripts/churn-prediction-app) for the full reasoning.
- Input validation currently checks types and required fields via Pydantic; range constraints (e.g., tenure ≥ 0) are a natural next addition.
