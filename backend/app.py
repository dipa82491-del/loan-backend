# =========================
# 1. IMPORT LIBRARIES
# =========================
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
import os

# =========================
# 2. LOAD MODEL & SCALER
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "loan_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# =========================
# 3. CREATE APP
# =========================
app = FastAPI(title="Loan Approval Prediction API")

# =========================
# 4. INPUT MODEL (VALIDATION)
# =========================
class LoanRequest(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int
    Property_Area: int

# =========================
# 5. ROOT
# =========================
@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

# =========================
# 6. PREDICTION
# =========================
@app.post("/predict")
def predict(data: LoanRequest):

    # =========================
    # INPUT VALIDATION
    # =========================
    if data.ApplicantIncome < 0 or data.CoapplicantIncome < 0:
        return {"status": "error", "message": "Income cannot be negative"}

    if data.LoanAmount <= 0:
        return {"status": "error", "message": "Loan amount must be greater than 0"}

    if data.Credit_History not in [0, 1]:
        return {"status": "error", "message": "Credit history must be 0 or 1"}

    # =========================
    # CONVERT INPUT
    # =========================
    input_data = np.array([[ 
        data.Gender,
        data.Married,
        data.Dependents,
        data.Education,
        data.Self_Employed,
        data.ApplicantIncome,
        data.CoapplicantIncome,
        data.LoanAmount,
        data.Loan_Amount_Term,
        data.Credit_History,
        data.Property_Area
    ]])

    # Apply scaling
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    result = "Approved" if prediction == 1 else "Rejected"

    # =========================
    # EXPLAINABLE AI / REJECTION ANALYSIS
    # =========================
    reasons = []

    total_income = data.ApplicantIncome + data.CoapplicantIncome
    debt_to_income_ratio = data.LoanAmount / total_income if total_income > 0 else float("inf")
    self_employed_unstable = data.Self_Employed == 1 and total_income < 3500

    if data.Credit_History == 0:
        reasons.append("Poor Credit History")

    if total_income < 2500:
        reasons.append("Income below minimum threshold")

    if debt_to_income_ratio > 40:
        reasons.append("High debt-to-income ratio")

    if self_employed_unstable:
        reasons.append("Self-employed stability risk")

    if prediction == 0:
        if not reasons:
            reasons.append("Model predicted rejection based on learned risk patterns")
    else:
        reasons.append("Eligible for Loan")
        if data.Credit_History == 1:
            reasons.append("Good Credit History")
        if total_income >= 2500 and debt_to_income_ratio <= 40:
            reasons.append("Healthy income and debt profile")
        if data.Self_Employed == 1 and not self_employed_unstable:
            reasons.append("Stable self-employed income")

    return {
        "status": "success",
        "prediction": result,
        "reasons": reasons
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)