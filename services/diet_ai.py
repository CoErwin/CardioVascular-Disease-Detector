import numpy as np
import os
import pickle

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "diet_model.pkl")

# ── Unhealthy food keywords to always exclude ──
UNHEALTHY_KEYWORDS = [
    "pizza", "burger", "mcdonalds", "dominos",
    "kfc", "fries", "sausage", "pepperoni",
    "hot dog", "hotdog", "bacon", "nuggets",
    "donut", "doughnut", "cookie", "brownie",
    "candy", "soda", "chips", "nacho",
    "milkshake", "waffle", "pancake syrup"
]

# ── High risk nutrition limits ──
HIGH_RISK_LIMITS = {
    "caloric value" : 400,   # max calories per item
    "fat"           : 15,    # max fat per item
}

def is_healthy(row, risk: float) -> bool:
    food_name = str(row.get("food", "")).lower()

    # Block unhealthy keywords
    for keyword in UNHEALTHY_KEYWORDS:
        if keyword in food_name:
            return False

    # For high risk users apply stricter nutrition limits
    if risk > 60:
        if float(row.get("caloric value", 0)) > HIGH_RISK_LIMITS["caloric value"]:
            return False
        if float(row.get("fat", 0)) > HIGH_RISK_LIMITS["fat"]:
            return False

    return True

def risk_to_nutrition_target(risk: float) -> dict:
    if risk < 30:
        return {
            "caloric value" : 600,
            "protein"       : 40,
            "carbohydrates" : 60,
            "fat"           : 20,
            "label"         : "Performance"
        }
    elif risk < 60:
        return {
            "caloric value" : 450,
            "protein"       : 30,
            "carbohydrates" : 50,
            "fat"           : 15,
            "label"         : "Balanced"
        }
    else:
        return {
            "caloric value" : 300,
            "protein"       : 20,
            "carbohydrates" : 35,
            "fat"           : 8,
            "label"         : "Heart-Healthy"
        }

def format_food(row) -> dict:
    return {
        "food"     : row.get("food", "Unknown"),
        "calories" : round(float(row.get("caloric value", 0)), 1),
        "protein"  : round(float(row.get("protein", 0)), 1),
        "carbs"    : round(float(row.get("carbohydrates", 0)), 1),
        "fat"      : round(float(row.get("fat", 0)), 1),
    }

def get_diet_recommendation(data: dict):
    try:
        risk      = float(data.get("risk", 50))
        diet_pref = str(data.get("diet_preference", "veg")).lower()
        target    = risk_to_nutrition_target(risk)

        # ── Load Model ──
        with open(MODEL_PATH, "rb") as f:
            saved = pickle.load(f)

        knn          = saved["model"]
        df           = saved["df"]
        scaler       = saved["scaler"]
        feature_cols = saved["feature_cols"]

        # ── Step 1: Filter by veg/non-veg ──
        if diet_pref in ["veg", "non-veg"]:
            df_filtered = df[df["diet_type"] == diet_pref].copy()
        else:
            df_filtered = df.copy()

        # ── Step 2: Filter out unhealthy foods ──
        df_filtered = df_filtered[
            df_filtered.apply(lambda row: is_healthy(row, risk), axis=1)
        ].copy()

        print(f"✅ Foods after filtering: {len(df_filtered)}")

        # ── Step 3: KNN on filtered dataset ──
        query_raw    = np.array([[
            target["caloric value"],
            target["protein"],
            target["carbohydrates"],
            target["fat"]
        ]])
        query_scaled = scaler.transform(query_raw)

        from sklearn.neighbors import NearestNeighbors
        X_filtered   = scaler.transform(df_filtered[feature_cols].values)
        knn_filtered = NearestNeighbors(
            n_neighbors=min(10, len(df_filtered)),
            metric="euclidean"
        )
        knn_filtered.fit(X_filtered)

        _, indices = knn_filtered.kneighbors(query_scaled)
        matches    = df_filtered.iloc[indices[0]]
        items      = [format_food(row) for _, row in matches.iterrows()]

        return {
            "diet_type"       : target["label"],
            "diet_preference" : diet_pref,
            "breakfast"       : items[0:2],
            "lunch"           : items[2:4],
            "dinner"          : items[4:6],
        }

    except Exception as e:
        print("❌ Diet ML Error:", e)
        return {
            "breakfast" : [{"food": "Oats",            "calories": 150, "protein": 5,  "carbs": 27, "fat": 3}],
            "lunch"     : [{"food": "Grilled Chicken",  "calories": 300, "protein": 35, "carbs": 0,  "fat": 7}],
            "dinner"    : [{"food": "Salad",            "calories": 200, "protein": 4,  "carbs": 15, "fat": 5}],
        }

