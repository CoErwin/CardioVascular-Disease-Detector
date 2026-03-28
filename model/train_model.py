import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/combined_dataset.csv")

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

# Drop unnecessary columns
df = df.dropna()

# Target column
target_col = "target"

X = df.drop(columns=[target_col])
y = df[target_col]

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= SCALE =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= MODEL =================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ================= EVALUATE =================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ================= SAVE =================
MODEL_PATH = os.path.join(BASE_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model + Scaler saved successfully")