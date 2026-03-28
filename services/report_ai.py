import pickle
import numpy as np
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

model  = pickle.load(open(MODEL_PATH,  "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# ── Healthy reference ranges ──
HEALTHY_RANGES = {
    "age"     : (0,   45,   "Age above 45 increases cardiovascular risk"),
    "trestbps": (0,   120,  "Resting blood pressure above 120 is elevated"),
    "chol"    : (0,   200,  "Cholesterol above 200 mg/dl is borderline high"),
    "thalach" : (150, 999,  "Maximum heart rate below 150 may indicate poor fitness"),
    "oldpeak" : (0,   1.0,  "ST depression above 1.0 indicates cardiac stress"),
    "ca"      : (0,   0,    "Any blocked vessels increases risk significantly"),
    "fbs"     : (0,   0,    "Fasting blood sugar above 120 mg/dl indicates diabetes risk"),
    "exang"   : (0,   0,    "Exercise induced angina detected"),
}

def get_risk_level(risk: float) -> dict:
    if risk > 70:
        return {
            "level"  : "High",
            "emoji"  : "🔴",
            "advice" : "High risk: Strict lifestyle change needed. Avoid salt, fried food, stress.",
            "doctor" : "⚠️ Strongly recommended to consult a cardiologist immediately."
        }
    elif risk > 40:
        return {
            "level"  : "Moderate",
            "emoji"  : "🟡",
            "advice" : "Moderate risk: Improve diet, exercise regularly.",
            "doctor" : "✔️ Consider doctor consultation within a few weeks."
        }
    else:
        return {
            "level"  : "Low",
            "emoji"  : "🟢",
            "advice" : "Low risk: Maintain healthy lifestyle.",
            "doctor" : "✔️ No immediate doctor visit required."
        }

def get_contributing_factors(data: dict) -> list:
    """
    Uses model feature importances × user values
    to find which features contributed most to THIS user's risk.
    """
    importances = model.feature_importances_
    features    = np.array([data.get(f, 0) for f in FEATURE_NAMES])
    scaled      = scaler.transform([features])[0]

    # Contribution = importance × |scaled value|
    contributions = importances * np.abs(scaled)
    ranked        = np.argsort(contributions)[::-1]

    top_factors = []
    for i in ranked[:4]:  # top 4 contributors
        fname = FEATURE_NAMES[i]
        value = data.get(fname, 0)
        top_factors.append({
            "feature"    : fname.upper(),
            "value"      : value,
            "importance" : round(float(importances[i]) * 100, 1)
        })

    return top_factors

def get_warnings(data: dict) -> list:
    """
    Compares user values against healthy ranges
    and returns personalized warnings.
    """
    warnings = []
    for feature, (low, high, message) in HEALTHY_RANGES.items():
        value = data.get(feature, None)
        if value is None:
            continue
        if feature in ["ca", "fbs", "exang"]:
            if value > high:
                warnings.append(message)
        elif feature == "thalach":
            if value < low:
                warnings.append(message)
        else:
            if value > high:
                warnings.append(message)
    return warnings

def get_report(data: dict) -> dict:
    try:
        risk        = float(data.get("risk", 0))
        risk_info   = get_risk_level(risk)
        factors     = get_contributing_factors(data)
        warnings    = get_warnings(data)

        # ── Personalized precautions based on risk level ──
        base_precautions = [
            "Exercise 30 mins daily",
            "Maintain healthy weight",
            "Monitor blood pressure regularly",
            "Avoid smoking and alcohol",
            "Get regular health checkups",
        ]

        if risk > 70:
            extra = [
                "Strictly reduce salt intake",
                "Avoid all fried & processed foods",
                "Reduce stress with meditation or yoga",
                "Take prescribed medications regularly",
            ]
        elif risk > 40:
            extra = [
                "Reduce salt and sugar intake",
                "Limit fried foods",
                "Consider stress management techniques",
            ]
        else:
            extra = [
                "Maintain balanced diet",
                "Stay hydrated",
            ]

        precautions = base_precautions + extra

        return {
            "risk"        : risk,
            "risk_level"  : risk_info["level"],
            "emoji"       : risk_info["emoji"],
            "advice"      : risk_info["advice"],
            "doctor"      : risk_info["doctor"],
            "precautions" : precautions,
            "top_factors" : factors,
            "warnings"    : warnings,
            "insight"     : f"Your top contributing factors were "
                            f"{factors[0]['feature']}, {factors[1]['feature']}, "
                            f"and {factors[2]['feature']} based on ML analysis."
        }

    except Exception as e:
        print("❌ Report Error:", e)
        return {
            "risk"        : data.get("risk", 0),
            "risk_level"  : "Unknown",
            "advice"      : "Please consult a doctor.",
            "doctor"      : "Consult a cardiologist.",
            "precautions" : ["Maintain healthy lifestyle"],
            "top_factors" : [],
            "warnings"    : [],
            "insight"     : "Could not generate ML insight."
        }