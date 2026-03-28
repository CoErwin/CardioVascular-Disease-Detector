import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "workout", "megaGymDataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "workout_model.pkl")

# ── Load & Clean ──
df = pd.read_csv(DATA_PATH)
df.columns = [col.lower().strip() for col in df.columns]
df = df.dropna()

# ── Encode Features ──
le_type  = LabelEncoder()
le_level = LabelEncoder()

df["type_enc"]  = le_type.fit_transform(df["type"])
df["level_enc"] = le_level.fit_transform(df["level"])

features = df[["type_enc", "level_enc"]].values

# ── Train KNN ──
knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
knn.fit(features)

print("✅ KNN model trained")
print(f"   Types  : {list(le_type.classes_)}")
print(f"   Levels : {list(le_level.classes_)}")

# ── Save Everything ──
with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model"   : knn,
        "df"      : df,
        "le_type" : le_type,
        "le_level": le_level,
    }, f)

print(f"✅ Saved → {MODEL_PATH}")