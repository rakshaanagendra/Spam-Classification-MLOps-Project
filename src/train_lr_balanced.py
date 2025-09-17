import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Load dataset
data = pd.read_csv(r"C:\Users\nagen\Desktop\Kaggle_datasets\SMS Spam collection\Spam-Classification-MLOps-Project\data\smsspamcollection\SMSSpamCollection",
                   sep="\t", header=None, names=["label", "message"])


X = data['message']
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression with balanced class weights
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Log to MLflow
with mlflow.start_run():
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("ngram_range", "1,2")
    mlflow.log_param("model", "LogisticRegression_balanced")

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, pos_label="spam"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, pos_label="spam"))
    mlflow.log_metric("f1", f1_score(y_test, y_pred, pos_label="spam"))

    mlflow.sklearn.log_model(model, "spam_model_lr_balanced")
