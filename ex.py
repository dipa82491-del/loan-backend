import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("/content/hotel_bookings.csv")

data = data.drop(["company", "agent", "reservation_status", "reservation_status_date"], axis=1)
data = data.fillna(0)

le = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    
    data[col] = data[col].astype(str)
    data[col] = le.fit_transform(data[col])

X = data.drop("is_canceled", axis=1)
y = data["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, solver='liblinear') 
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))