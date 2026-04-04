# =========================
# 1. IMPORT LIBRARIES
# =========================
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
<<<<<<< HEAD
=======
import os
>>>>>>> 4263363 (Updated dataset + model + FastAPI app)

# =========================
# 2. LOAD MODEL & SCALER
# =========================
<<<<<<< HEAD
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
=======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "loan_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
>>>>>>> 4263363 (Updated dataset + model + FastAPI app)

# =========================
# 3. CREATE APP
# =========================
<<<<<<< HEAD
app = FastAPI()

# =========================
# 4. INPUT FORMAT (IMPORTANT)
=======
app = FastAPI(title="Loan Approval Prediction API")

# =========================
# 4. INPUT MODEL (VALIDATION)
>>>>>>> 4263363 (Updated dataset + model + FastAPI app)
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
<<<<<<< HEAD
    Credit_History: float
    Property_Area: int

# =========================
# 5. ROOT CHECK
=======
    Credit_History: int
    Property_Area: int

# =========================
# 5. ROOT
>>>>>>> 4263363 (Updated dataset + model + FastAPI app)
# =========================
@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

# =========================
<<<<<<< HEAD
# 6. PREDICTION API
=======
# 6. PREDICTION
>>>>>>> 4263363 (Updated dataset + model + FastAPI app)
# =========================
@app.post("/predict")
def predict(data: LoanRequest):

<<<<<<< HEAD
    # Convert input to array
    input_data = np.array([[
=======
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
>>>>>>> 4263363 (Updated dataset + model + FastAPI app)
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

<<<<<<< HEAD
    # Apply scaling
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Output
    result = "Approved" if prediction == 1 else "Rejected"

    return {"prediction": result}


=======
    # =========================
    # SCALE + PREDICT
    # =========================
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
    except Exception as e:
        return {"status": "error", "message": str(e)}

    result = "Approved" if prediction == 1 else "Rejected"

    # =========================
    # EXPLAINABLE AI (IMPROVED)
    # =========================
    reasons = []
    total_income = data.ApplicantIncome + data.CoapplicantIncome

    # Rejection logic aligned with dataset
    if prediction == 0:
        if total_income < 3000:
            reasons.append("Low Income")

        if data.LoanAmount > 300:
            reasons.append("High Loan Amount")

        if data.Credit_History == 0:
            reasons.append("Poor Credit History")

    else:
        reasons.append("Eligible for Loan")

        if data.Credit_History == 1:
            reasons.append("Good Credit History")

        if total_income > 5000:
            reasons.append("Strong Income")

    # =========================
    # RESPONSE
    # =========================
    return {
        "status": "success",
        "prediction": result,
        "reasons": reasons
    }

# =========================
# 7. RUN SERVER
# =========================
>>>>>>> 4263363 (Updated dataset + model + FastAPI app)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)