import pandas as pd

# ==========================
# 1️⃣ Define column names
# ==========================
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# ==========================
# 2️⃣ Load UCI raw datasets
# ==========================
cleveland = pd.read_csv("processed.cleveland.data", names=columns)
hungarian = pd.read_csv("processed.hungarian.data", names=columns)
switzerland = pd.read_csv("processed.switzerland.data", names=columns)
va = pd.read_csv("processed.va.data", names=columns)

# ==========================
# 3️⃣ Combine UCI datasets
# ==========================
uci_combined = pd.concat(
    [cleveland, hungarian, switzerland, va],
    axis=0
)

print("UCI Combined Shape:", uci_combined.shape)

# ==========================
# 4️⃣ Clean UCI data
# ==========================

# Replace '?' with NaN
uci_combined.replace('?', pd.NA, inplace=True)

# Drop missing rows
uci_combined.dropna(inplace=True)

# Convert to numeric
uci_combined = uci_combined.astype(float)

# Convert target >0 → 1
uci_combined['target'] = uci_combined['target'].apply(
    lambda x: 1 if x > 0 else 0
)

print("UCI Cleaned Shape:", uci_combined.shape)

# ==========================
# 5️⃣ Load Kaggle datasets
# ==========================
kaggle1 = pd.read_csv("cleaned_merged_heart_dataset.csv")
statlog = pd.read_csv("Heart_disease_statlog.csv")

# ==========================
# 6️⃣ Standardize column names
# ==========================
kaggle1.columns = columns
statlog.columns = columns

# Convert target >0 → 1
kaggle1['target'] = kaggle1['target'].apply(lambda x: 1 if x > 0 else 0)
statlog['target'] = statlog['target'].apply(lambda x: 1 if x > 0 else 0)

# ==========================
# 7️⃣ Merge Everything
# ==========================
final_combined = pd.concat(
    [uci_combined, kaggle1, statlog],
    axis=0
)

# Shuffle dataset
final_combined = final_combined.sample(frac=1, random_state=42)

print("Final Dataset Shape:", final_combined.shape)

# ==========================
# 8️⃣ Save combined_dataset.csv
# ==========================
final_combined.to_csv("combined_dataset.csv", index=False)

print("combined_dataset.csv created successfully!")