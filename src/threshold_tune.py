# src/threshold_tune.py
"""
Train a logistic regression spam classifier, sweep thresholds, pick best by F1,
and save model+vectorizer+threshold to ../outputs.

Run from project root:
    python .\src\threshold_tune.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Optional MLflow logging (won't crash if mlflow missing)
try:
    import mlflow
    mlflow_installed = True
except Exception:
    mlflow_installed = False

# -------------------------
# Paths (relative, robust)
# -------------------------
ROOT = os.path.dirname(__file__)         # src/
DATA_PATH = os.path.join(ROOT, "..", "data", "smsspamcollection", "SMSSpamCollection")
OUTPUT_DIR = os.path.join(ROOT, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load data
# -------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}\n"
                            "Make sure the dataset exists at this relative path.")

data = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
# ensure labels are strings like 'spam'/'ham'
data['label'] = data['label'].astype(str)

# -------------------------
# Train/test split
# -------------------------
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Vectorize text
# (adjust max_features if you want)
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------
# Train model
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -------------------------
# Get predicted probabilities (fixes y_proba NameError)
# -------------------------
# model.classes_ holds the class labels order used by predict_proba columns
# find index of "spam" class in that order (safe even if labels different order)
if "spam" in list(model.classes_):
    spam_index = list(model.classes_).index("spam")
else:
    # fallback: assume positive class is the second column (index 1)
    spam_index = 1

y_proba = model.predict_proba(X_test_tfidf)  # shape (n_samples, n_classes)
probs_spam = y_proba[:, spam_index]          # probability for 'spam' class

# -------------------------
# Threshold sweep
# -------------------------
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

print("threshold | precision | recall | f1 | tp fp fn tn")
for t in thresholds:
    # predicted labels at threshold t (strings 'spam'/'ham')
    y_pred_custom = np.where(probs_spam >= t, "spam", "ham")

    precision = precision_score(y_test, y_pred_custom, pos_label="spam", zero_division=0)
    recall = recall_score(y_test, y_pred_custom, pos_label="spam", zero_division=0)
    f1 = f1_score(y_test, y_pred_custom, pos_label="spam", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_custom, labels=["ham", "spam"]).ravel()

    results.append({
        "threshold": t,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    })

    print(f" {t:6.1f} | {precision:0.3f}    | {recall:0.3f}  | {f1:0.3f} | {tp:3d} {fp:3d}  {fn:3d} {tn:3d}")

# -------------------------
# Choose best threshold by F1 (tie-breaker: higher precision)
# -------------------------
best = None
for r in results:
    if best is None:
        best = r
        continue
    if r["f1"] > best["f1"]:
        best = r
    elif r["f1"] == best["f1"] and r["precision"] > best["precision"]:
        best = r

best_threshold = best["threshold"]
best_f1 = best["f1"]
print(f"\n✅ Best threshold selected: {best_threshold} (F1={best_f1:0.3f})")

# -------------------------
# Save artifacts
# -------------------------
joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))
joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "vectorizer.pkl"))

with open(os.path.join(OUTPUT_DIR, "threshold.json"), "w") as fh:
    json.dump({"threshold": float(best_threshold)}, fh)

print(f"✅ Saved model.pkl, vectorizer.pkl, threshold.json to {OUTPUT_DIR}")

# -------------------------
# Optional: MLflow logging
# -------------------------
if mlflow_installed:
    try:
        with mlflow.start_run():
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_features", 5000)
            mlflow.log_metric("best_threshold", float(best_threshold))
            mlflow.log_metric("best_f1", float(best_f1))
            mlflow.sklearn.log_model(model, artifact_path="spam_model")
            print("✅ Logged model & metrics to MLflow")
    except Exception as e:
        print("⚠️ MLflow logging failed:", e)
