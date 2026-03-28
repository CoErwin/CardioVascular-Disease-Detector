import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

# ── Paths ──
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, "../data/combined_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "heart_model.pkl")
SCALER_PATH= os.path.join(BASE_DIR, "scaler.pkl")
OUT_DIR    = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ──
model  = pickle.load(open(MODEL_PATH,  "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

df = pd.read_csv(DATA_PATH).dropna()
X  = df.drop(columns=["target"])
y  = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_test_scaled = scaler.transform(X_test)

y_pred  = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ════════════════════════════════════════════
# 1. CONFUSION MATRIX
# ════════════════════════════════════════════
def plot_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "1_confusion_matrix.png"), dpi=150)
    plt.close()
    print("✅ Saved: 1_confusion_matrix.png")

# ════════════════════════════════════════════
# 2. ROC CURVE
# ════════════════════════════════════════════
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
             label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "2_roc_curve.png"), dpi=150)
    plt.close()
    print("✅ Saved: 2_roc_curve.png")

# ════════════════════════════════════════════
# 3. FEATURE IMPORTANCE
# ════════════════════════════════════════════
def plot_feature_importance():
    importances = model.feature_importances_
    features    = X.columns
    indices     = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features)), importances[indices], color="steelblue")
    plt.xticks(range(len(features)),
               [features[i] for i in indices], rotation=45, ha="right")
    plt.title("Feature Importance", fontsize=14, fontweight="bold")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "3_feature_importance.png"), dpi=150)
    plt.close()
    print("✅ Saved: 3_feature_importance.png")

# ════════════════════════════════════════════
# 4. PRECISION-RECALL CURVE
# ════════════════════════════════════════════
def plot_precision_recall():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="green", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "4_precision_recall.png"), dpi=150)
    plt.close()
    print("✅ Saved: 4_precision_recall.png")

# ════════════════════════════════════════════
# 5. CROSS-VALIDATION SCORES
# ════════════════════════════════════════════
def plot_cross_validation():
    X_scaled = scaler.transform(X)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

    plt.figure(figsize=(6, 5))
    bars = plt.bar([f"Fold {i+1}" for i in range(5)],
                   cv_scores * 100, color="mediumpurple")
    plt.axhline(y=cv_scores.mean() * 100, color="red",
                linestyle="--", label=f"Mean: {cv_scores.mean()*100:.2f}%")
    for bar, score in zip(bars, cv_scores):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{score*100:.1f}%", ha="center", fontsize=10)
    plt.title("5-Fold Cross Validation Accuracy",
              fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.ylim(80, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "5_cross_validation.png"), dpi=150)
    plt.close()
    print("✅ Saved: 5_cross_validation.png")

# ════════════════════════════════════════════
# 6. CLASSIFICATION REPORT HEATMAP
# ════════════════════════════════════════════
def plot_classification_report():
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().iloc[:4, :3]

    plt.figure(figsize=(7, 4))
    sns.heatmap(report_df, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5)
    plt.title("Classification Report Heatmap",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "6_classification_report.png"), dpi=150)
    plt.close()
    print("✅ Saved: 6_classification_report.png")

# ════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n📊 Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
    plot_confusion_matrix()
    plot_roc_curve()
    plot_feature_importance()
    plot_precision_recall()
    plot_cross_validation()
    plot_classification_report()
    print(f"\n✅ All plots saved → backend/model/plots/")