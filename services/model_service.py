import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))


def predict_risk(data: dict):
    try:
        features = np.array([[
            data["age"],
            data["sex"],
            data["cp"],
            data["trestbps"],
            data["chol"],
            data["fbs"],
            data["restecg"],
            data["thalach"],
            data["exang"],
            data["oldpeak"],
            data["slope"],
            data["ca"],
            data["thal"]
        ]])

        scaled = scaler.transform(features)

        prob = model.predict_proba(scaled)[0][1]

        risk = float(round(prob * 100, 2))  # ✅ ALWAYS NUMBER

        return {"risk": risk}  # ✅ STANDARD KEY

    except Exception as e:
        print("❌ Prediction Error:", e)
        return {"risk": 0.0}