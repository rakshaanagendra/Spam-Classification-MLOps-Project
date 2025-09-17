# src/threshold_tune.py
# Purpose: load best TF-IDF + LR model (or retrain quickly), compute predicted probabilities,
# and evaluate precision/recall/f1 for several thresholds. Also plot Precision-Recall curve.

import pandas as pd                               # Data handling
import numpy as np                                # Numeric utilities
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt                   # plotting

# ---- Load dataset ----
# 1) read dataset into DataFrame
data = pd.read_csv(r"C:\Users\nagen\Desktop\Kaggle_datasets\SMS Spam collection\Spam-Classification-MLOps-Project\data\smsspamcollection\SMSSpamCollection",
                   sep="\t", header=None, names=["label", "message"])

# Explanation: read_csv reads the file; sep="\t" because the file is tab-separated; header=None since file has no header row.

# ---- Train/test split ----
X = data["message"]   # feature column: raw text messages
y = data["label"]     # target: 'ham' or 'spam'
# Explanation: separate features and labels into X and y.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Explanation:
# - test_size=0.2 -> keep 20% for final evaluation.
# - random_state=42 -> reproducible split.
# - stratify=y -> preserve spam/ham ratio in train and test.

# ---- Vectorize and train a Logistic Regression (use tuned C) ----
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))
# Explanation: TF-IDF vectorizer that removes English stopwords, keeps top 10k features, and includes bigrams.

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Explanation: fit_transform learns vocabulary from training and converts train text to sparse matrix;
# transform converts test text using same vocabulary.

model = LogisticRegression(max_iter=1000, C=10, class_weight="balanced")
# Explanation: Logistic Regression using best C=10 from GridSearch and class_weight balanced to mitigate imbalance.
model.fit(X_train_tfidf, y_train)
# Explanation: trains (fits) the model on training features and labels.

# ---- Get predicted probabilities for the "spam" class ----
# sklearn's predict_proba returns probabilities for each class in label order. We find the column corresponding to "spam".
probas = model.predict_proba(X_test_tfidf)  # shape (n_samples, n_classes)
# Explanation: predict_proba gives probabilities for each class (ordered by model.classes_).

# find index of spam class
spam_index = list(model.classes_).index("spam")
spam_scores = probas[:, spam_index]
# Explanation: choose the probability for the "spam" class for each sample.

# ---- Precision-Recall curve ----
precision, recall, thresholds = precision_recall_curve(y_test == "spam", spam_scores)
# Explanation: precision_recall_curve expects binary true values (True for spam), and continuous scores.
# It returns arrays of precision, recall and the corresponding thresholds.

# Plot Precision-Recall curve
plt.figure(figsize=(6,4))
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.grid(True)
plt.show()
# Explanation: visualizes trade-off between precision and recall as threshold changes.

# ---- Try a few thresholds and print metrics ----
candidate_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("threshold | precision | recall | f1 | tp fp fn tn")
for t in candidate_thresholds:
    preds = np.where(spam_scores >= t, "spam", "ham")
    p = precision_score(y_test == "spam", preds == "spam")
    r = recall_score(y_test == "spam", preds == "spam")
    f1 = f1_score(y_test == "spam", preds == "spam")
    cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]] but with our labels: rows = true, cols = pred
    tn, fp = cm[0,0], cm[0,1]
    fn, tp = cm[1,0], cm[1,1]
    print(f"{t:>8} | {p:.3f}     | {r:.3f}  | {f1:.3f} | {tp:3d} {fp:3d} {fn:3d} {tn:3d}")

# Explanation:
# - For each threshold, we derive predictions as spam if prob >= t.
# - We compute precision, recall, f1 using binary comparisons (spam True).
# - We print confusion numbers to see trade-off (tp, fp, fn, tn).

# ---- To choose a threshold ----
# Inspect the printed lines and the PR-curve. Choose a threshold that meets your preference:
# - If you want very high recall (catch most spam), pick a lower threshold (e.g., 0.3-0.45).
# - If you want few false positives, pick a higher threshold.
