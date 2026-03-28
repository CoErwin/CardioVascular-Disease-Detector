import pickle
import pandas as pd
import os  # ← add this
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Fix paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/combined_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load model & scaler
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# Load dataset
df = pd.read_csv(DATA_PATH).dropna()
X = df.drop(columns=["target"])
y = df["target"]

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))