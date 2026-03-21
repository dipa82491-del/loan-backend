# =========================
# 1. IMPORT LIBRARIES
# =========================
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# =========================
# 2. LOAD MODEL & SCALER
# =========================
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# 3. CREATE APP
# =========================
app = FastAPI()

# =========================
# 4. INPUT FORMAT (IMPORTANT)
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
    Credit_History: float
    Property_Area: int

# =========================
# 5. ROOT CHECK
# =========================
@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

# =========================
# 6. PREDICTION API
# =========================
@app.post("/predict")
def predict(data: LoanRequest):

    # Convert input to array
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

    # Output
    result = "Approved" if prediction == 1 else "Rejected"

    return {"prediction": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)