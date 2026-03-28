import pandas as pd
import os

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "food", "final_food_dataset.csv")
OUT_PATH  = os.path.join(BASE_DIR, "data", "food", "final_food_dataset_tagged.csv")

# ── Common non-veg keywords ──
NON_VEG_KEYWORDS = [
    "chicken", "beef", "pork", "lamb", "fish", "shrimp", "prawn",
    "salmon", "tuna", "crab", "lobster", "turkey", "bacon", "sausage",
    "mutton", "duck", "veal", "anchovy", "sardine", "meat", "steak",
    "ham", "pepperoni", "salami", "meatball", "kebab", "egg"
]

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()

def tag_veg(food_name: str) -> str:
    name = str(food_name).lower()
    for keyword in NON_VEG_KEYWORDS:
        if keyword in name:
            return "non-veg"
    return "veg"

df["diet_type"] = df["food"].apply(tag_veg)

df.to_csv(OUT_PATH, index=False)

print(f"✅ Tagged dataset saved → {OUT_PATH}")
print(df["diet_type"].value_counts())