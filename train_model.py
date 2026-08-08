# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("parkinsons.csv")

print("First 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

# -----------------------------
# Separate Features and Target
# -----------------------------

# Remove the 'name' column
X = data.drop(columns=["name", "status"])

# Target column
y = data["status"]

# -----------------------------
# Handle Missing Values
# -----------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Test Model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Files
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("\nFiles Saved Successfully!")
print("model.pkl")
print("scaler.pkl")
print("imputer.pkl")