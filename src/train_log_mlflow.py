import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# ========================
#  Load Data
# ========================
DATA_PATH = "data/smsspamcollection/SMSSpamCollection"
df = pd.read_csv(DATA_PATH, sep="\t", names=["label", "message"])

# Encode labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ========================
#  Feature Engineering
# ========================
def add_features(texts):
    return np.array([
        [len(t), sum(c.isdigit() for c in t), sum(c.isupper() for c in t), sum(c in "!?.,;" for c in t)]
        for t in texts
    ])

feature_engineer = FunctionTransformer(add_features)

preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(max_features=3000), "message"),
        ("features", feature_engineer, "message"),
    ],
    remainder="drop"
)

# ========================
#  Pipeline + Training
# ========================
pipeline = Pipeline([
    ("features", preprocessor),
    ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42))
])

pipeline.fit(pd.DataFrame({"message": X_train}), y_train)
y_prob = pipeline.predict_proba(pd.DataFrame({"message": X_test}))[:, 1]

# ========================
#  MLflow Logging
# ========================
mlflow.set_experiment("SpamClassification")

with mlflow.start_run(run_name="LogReg_with_thresholds"):
    # Default threshold (0.5)
    default_thr = 0.5
    y_pred_default = (y_prob >= default_thr).astype(int)

    acc = accuracy_score(y_test, y_pred_default)
    prec = precision_score(y_test, y_pred_default)
    rec = recall_score(y_test, y_pred_default)
    f1 = f1_score(y_test, y_pred_default)

    print(f"Default Threshold (0.5) -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_default))

    mlflow.log_metric("acc_default", acc)
    mlflow.log_metric("prec_default", prec)
    mlflow.log_metric("rec_default", rec)
    mlflow.log_metric("f1_default", f1)

    # Threshold sweep
    thresholds = np.arange(0.1, 0.91, 0.05)
    f1_scores = []
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        f1_thr = f1_score(y_test, preds)
        f1_scores.append(f1_thr)
        mlflow.log_metric(f"f1_thr_{thr:.2f}", f1_thr)

    # Best threshold
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    mlflow.log_metric("best_thr", best_thr)
    mlflow.log_metric("best_f1", best_f1)

    print(f"Best threshold: {best_thr:.2f} with F1={best_f1:.4f}")

    # ========================
    #  Visualizations
    # ========================
    os.makedirs("outputs/assets", exist_ok=True)

    # Confusion matrix (best threshold)
    cm = confusion_matrix(y_test, (y_prob >= best_thr).astype(int))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Best Thr={best_thr:.2f})")
    plt.tight_layout()
    cm_path = "outputs/assets/conf_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Threshold vs F1 curve
    plt.figure(figsize=(6,4))
    plt.plot(thresholds, f1_scores, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold vs F1 Score")
    plt.grid(True)
    f1_path = "outputs/assets/threshold_f1.png"
    plt.savefig(f1_path)
    plt.close()
    mlflow.log_artifact(f1_path)

    # Save model
    mlflow.sklearn.log_model(pipeline, "SpamLR")
