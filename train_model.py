# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# =========================
# 2. LOAD DATA
# =========================
data = pd.read_csv("loan_dataset_500.csv")

print("Dataset Preview:\n", data.head())

# =========================
# 3. DATA PREPROCESSING
# =========================

# Convert categorical to numerical
data.replace({
    'Gender': {'Male':1, 'Female':0},
    'Married': {'Yes':1, 'No':0},
    'Education': {'Graduate':1, 'Not Graduate':0},
    'Self_Employed': {'Yes':1, 'No':0},
    'Property_Area': {'Urban':2, 'Semiurban':1, 'Rural':0},
    'Loan_Status': {'Approved':1, 'Rejected':0},
    'Dependents': {'0':0, '1':1, '2':2, '3+':3}
}, inplace=True)

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# =========================
# 4. FEATURE & TARGET
# =========================
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# =========================
# 5. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. FEATURE SCALING
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 7. MODEL TRAINING
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# 8. MODEL EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))

# =========================
# 9. SAVE MODEL
# =========================
pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel and scaler saved successfully!")

# =========================
# 10. PREDICTION FUNCTION
# =========================
def predict_loan(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        result = "Approved"
    else:
        result = "Rejected"
    
    # Explainable AI (Reasons)
    reasons = []
    
    applicant_income = input_data[0][5]
    loan_amount = input_data[0][7]
    credit_history = input_data[0][9]
    
    if applicant_income < 2500:
        reasons.append("Low Income")
    
    if loan_amount > 300:
        reasons.append("High Loan Amount")
    
    if credit_history == 0:
        reasons.append("Poor Credit History")
    
    return result, reasons

# =========================
# 11. SAMPLE TEST
# =========================
sample = [1, 1, 0, 1, 0, 5000, 2000, 150, 360, 1, 2]

result, reasons = predict_loan(sample)

print("\nPrediction:", result)
print("Reasons:", reasons)