import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "food", "final_food_dataset_tagged.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "diet_model.pkl")

# ── Load ──
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()
df = df.dropna()

# ── Encode diet_type ──
le = LabelEncoder()
df["diet_type_enc"] = le.fit_transform(df["diet_type"])  # veg=1, non-veg=0

# ── Features ──
feature_cols = ["caloric value", "protein", "carbohydrates", "fat"]
X = df[feature_cols].values

# ── Scale ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train KNN ──
knn = NearestNeighbors(n_neighbors=20, metric="euclidean")
knn.fit(X_scaled)

print("✅ Diet KNN retrained")
print(f"   Veg items    : {len(df[df['diet_type']=='veg'])}")
print(f"   Non-Veg items: {len(df[df['diet_type']=='non-veg'])}")

# ── Save ──
with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model"       : knn,
        "df"          : df,
        "scaler"      : scaler,
        "feature_cols": feature_cols,
        "le"          : le,
    }, f)

print(f"✅ Saved → {MODEL_PATH}")