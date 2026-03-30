import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

BASE_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/combined_dataset.csv")

# ════════════════════════════════════════════
# 1. LOAD & SPLIT
# ════════════════════════════════════════════
df = pd.read_csv(DATA_PATH).dropna()

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ════════════════════════════════════════════
# 2. SCALE
# ════════════════════════════════════════════
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ════════════════════════════════════════════
# 3. DEFINE BASE MODELS
# ════════════════════════════════════════════
rf = RandomForestClassifier(
    n_estimators = 200,
    max_depth    = 10,
    random_state = 42
)

xgb = XGBClassifier(
    n_estimators       = 300,
    max_depth          = 6,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    use_label_encoder  = False,
    eval_metric        = "logloss",
    random_state       = 42
)

gb = GradientBoostingClassifier(
    n_estimators  = 200,
    max_depth     = 5,
    learning_rate = 0.05,
    random_state  = 42
)

lr = LogisticRegression(
    max_iter     = 1000,
    random_state = 42
)

# ════════════════════════════════════════════
# 4. STACKING ENSEMBLE
# Meta-learner (Logistic Regression) learns
# how to best combine base model predictions
# ════════════════════════════════════════════
stacking_model = StackingClassifier(
    estimators=[
        ("rf",  rf),
        ("xgb", xgb),
        ("gb",  gb),
    ],
    final_estimator = LogisticRegression(max_iter=1000),
    cv              = 5,        # 5-fold CV for generating meta-features
    stack_method    = "predict_proba",
    passthrough     = False
)

# ════════════════════════════════════════════
# 5. TRAIN
# ════════════════════════════════════════════
print("⏳ Training stacking ensemble...")
stacking_model.fit(X_train_scaled, y_train)
print("✅ Training complete")

# ════════════════════════════════════════════
# 6. EVALUATE
# ════════════════════════════════════════════
y_pred   = stacking_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Individual Model Accuracies:")

for name, base_model in [("Random Forest", rf), ("XGBoost", xgb), ("Gradient Boost", gb)]:
    base_model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, base_model.predict(X_test_scaled))
    print(f"   {name:20s}: {acc * 100:.2f}%")

print(f"\n🏆 Stacking Ensemble Accuracy : {accuracy * 100:.2f}%")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# ════════════════════════════════════════════
# 7. CROSS VALIDATION ON FULL DATASET
# ════════════════════════════════════════════
print("⏳ Running 5-Fold Cross Validation...")
X_all_scaled = scaler.transform(X)
cv_scores    = cross_val_score(
    stacking_model, X_all_scaled, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy"
)
print(f"✅ CV Scores : {[round(s*100,2) for s in cv_scores]}")
print(f"✅ CV Mean   : {cv_scores.mean()*100:.2f}%")
print(f"✅ CV Std    : {cv_scores.std()*100:.2f}%")

# ════════════════════════════════════════════
# 8. SAVE
# ════════════════════════════════════════════
MODEL_PATH  = os.path.join(BASE_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(stacking_model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"\n✅ Stacking model + scaler saved successfully")
