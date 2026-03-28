import pandas as pd
import numpy as np
import os
import pickle

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "workout_model.pkl")

# ── Risk + Age → Fitness Level ──
def get_fitness_level(risk: float, age: int) -> str:
    if risk > 70 or age > 60:
        return "Beginner"
    elif risk > 40 or age > 45:
        return "Intermediate"
    else:
        return "Expert"

# ── Fitness Level → Sets/Reps/Duration ──
def level_to_plan(level: str) -> dict:
    config = {
        "Beginner"     : {"sets": 2, "reps": 8,  "duration": "15 mins"},
        "Intermediate" : {"sets": 3, "reps": 12, "duration": "20 mins"},
        "Expert"       : {"sets": 5, "reps": 15, "duration": "30 mins"},
    }
    return config.get(level, config["Intermediate"])

def get_workout_recommendation(user_data: dict):
    try:
        risk  = float(user_data.get("risk", 50))
        age   = int(user_data.get("age",   35))
        level = get_fitness_level(risk, age)

        print(f"✅ Risk: {risk}, Age: {age}, Level: {level}")

        # ── Load Model ──
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        knn      = data["model"]
        df       = data["df"]
        le_type  = data["le_type"]
        le_level = data["le_level"]

        type_map = {
            "cardio"      : "Cardio",
            "strength"    : "Strength",
            "flexibility" : "Stretching",
        }

        result = {}

        for category, csv_type in type_map.items():
            try:
                type_enc = le_type.transform([csv_type])[0]
            except ValueError:
                type_enc = 0

            try:
                level_enc = le_level.transform([level])[0]
            except ValueError:
                level_enc = 0

            query        = np.array([[type_enc, level_enc]])
            _, indices   = knn.kneighbors(query)

            recommendations = df.iloc[indices[0]]
            plan_config     = level_to_plan(level)

            plan = []
            for _, row in recommendations.head(5).iterrows():
                entry = {
                    "name"  : row.get("title", "Unknown"),
                    "level" : level,
                    "sets"  : plan_config["sets"],
                    "reps"  : plan_config["reps"],
                }
                if category == "cardio":
                    entry["duration"] = plan_config["duration"]
                plan.append(entry)

            result[category] = plan

        result["fitness_level"] = level  # ← send level to frontend too
        return result

    except Exception as e:
        print("❌ Workout ML Error:", e)
        return {
            "cardio"      : [{"name": "Walking",  "duration": "20 mins"}],
            "strength"    : [{"name": "Pushups",  "sets": 3, "reps": 10}],
            "flexibility" : [{"name": "Yoga",     "duration": "15 mins"}],
        }