from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib  # To load your trained model
import numpy as np

app = FastAPI()

# This matches your Android 'LoanRequest' data class exactly
class LoanRequest(BaseModel):
    income: float
    coapplicant_income: float
    loan_amount: float
    loan_term: int
    credit_history: int
    property_area: str

@app.get("/")
def home():
    return {"message": "Loan Prediction API is Running"}

@app.post("/predict")
def predict_loan(request: LoanRequest):
    # 1. Convert request to a Dictionary
    data = request.dict()
    
    # 2. FEATURE ENGINEERING (This is how you improve the ML model)
    # Banks care about total household income and how much of it goes to the loan
    total_income = data['income'] + data['coapplicant_income']
    
    # Estimate Monthly Installment (EMI)
    # Simple calculation: Loan / Months
    emi = data['loan_amount'] / data['loan_term'] if data['loan_term'] > 0 else 0
    
    # Debt-to-Income Ratio (Very important for banks!)
    dti_ratio = emi / (total_income / 12) if total_income > 0 else 0

    # 3. Prepare data for the model
    # Note: You should have a pre-trained model saved as 'loan_model.pkl'
    # For now, I'll show the logic for a simple Rule-Based check + ML call
    
    # Rule 1: Credit History is the strongest predictor
    if data['credit_history'] == 0:
        return {"prediction": "Rejected (Low Credit Score)"}

    # Rule 2: Extremely high debt-to-income ratio
    if dti_ratio > 0.5:
        return {"prediction": "Rejected (High Debt-to-Income)"}

    # 4. ML MODEL CALL (Example)
    # In a real scenario, you would do this:
    # model = joblib.load("loan_model.pkl")
    # input_df = pd.DataFrame([data]) 
    # prediction = model.predict(input_df)
    
    # Mocking a successful ML response based on Total Income vs Loan
    if total_income > (data['loan_amount'] * 0.1): # Simple logic for demonstration
        return {"prediction": "Approved"}
    else:
        return {"prediction": "Rejected (Insufficient Income)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)