import os
import pandas as pd
import numpy as np
import string
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_curve, auc, accuracy_score
)

# -----------------------
# 1. Load dataset
# -----------------------
DATA_PATH = "data/smsspamcollection/SMSSpamCollection"

df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# -----------------------
# 2. Feature engineering
# -----------------------
def add_features(df):
    return pd.DataFrame({
        "msg_len": df["message"].apply(len),
        "digit_count": df["message"].apply(lambda x: sum(c.isdigit() for c in x)),
        "punct_count": df["message"].apply(lambda x: sum(c in string.punctuation for c in x)),
        "upper_ratio": df["message"].apply(lambda x: sum(c.isupper() for c in x)) / df["message"].apply(len)
    })

X_train, X_test, y_train, y_test = train_test_split(
    df[["message"]], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# -----------------------
# 3. Pipeline
# -----------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=3000)

preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", tfidf, "message"),
        ("features", FunctionTransformer(add_features), ["message"])
    ],
    remainder="drop"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42))
])

# -----------------------
# 4. Train
# -----------------------
pipeline.fit(X_train, y_train)

# Probabilities for threshold tuning
y_proba = pipeline.predict_proba(X_test)[:, 1]

# -----------------------
# 5. Threshold sweep
# -----------------------
thresholds = np.linspace(0.1, 0.9, 9)
threshold_results = []
for t in thresholds:
    preds = (y_proba >= t).astype(int)
    threshold_results.append({
        "threshold": t,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    })

thresh_df = pd.DataFrame(threshold_results)

# Find best threshold (max F1)
best_row = thresh_df.loc[thresh_df["f1"].idxmax()]
best_threshold = best_row["threshold"]
best_f1 = best_row["f1"]

# Confusion matrix at best threshold
best_preds = (y_proba >= best_threshold).astype(int)
cm = confusion_matrix(y_test, best_preds)

print(f"Best threshold: {best_threshold:.2f} with F1={best_f1:.4f}")
print("Confusion matrix at best threshold:\n", cm)

# -----------------------
# 6. Curves
# -----------------------
precisions, recalls, thres = precision_recall_curve(y_test, y_proba)
fpr, tpr, roc_thres = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Save artifacts
os.makedirs("outputs", exist_ok=True)

plt.figure()
plt.plot(recalls, precisions, marker=".")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("outputs/pr_curve.png")
plt.close()

plt.figure()
plt.plot(fpr, tpr, marker=".")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
plt.savefig("outputs/roc_curve.png")
plt.close()

thresh_df.to_csv("outputs/threshold_metrics.csv", index=False)
pd.DataFrame(cm).to_csv("outputs/confusion_matrix_best_thr.csv", index=False)

# -----------------------
# 7. MLflow logging
# -----------------------
mlflow.set_experiment("SpamClassification")

with mlflow.start_run(run_name="LogReg_C10_Balanced"):
    # Params
    mlflow.log_param("C", 10)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("max_features", 3000)

    # Metrics for ALL thresholds
    for _, row in thresh_df.iterrows():
        thr = row["threshold"]
        mlflow.log_metric(f"accuracy_thr_{thr:.2f}", row["accuracy"])
        mlflow.log_metric(f"precision_thr_{thr:.2f}", row["precision"])
        mlflow.log_metric(f"recall_thr_{thr:.2f}", row["recall"])
        mlflow.log_metric(f"f1_thr_{thr:.2f}", row["f1"])

    # ROC AUC
    mlflow.log_metric("roc_auc", roc_auc)

    # Best threshold + F1
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_f1", best_f1)

    # Model
    mlflow.sklearn.log_model(pipeline, "model", registered_model_name="SpamLR")

    # Artifacts
    mlflow.log_artifacts("outputs")

print("✅ Training complete. Metrics for all thresholds, best threshold/F1, and confusion matrix logged to MLflow.")
