import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# 1. Load dataset
data = pd.read_csv(r"C:\Users\nagen\Desktop\Kaggle_datasets\SMS Spam collection\Spam-Classification-MLOps-Project\data\smsspamcollection\SMSSpamCollection",
                   sep="\t", header=None, names=["label", "message"])

X = data['message']
y = data['label']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorizer with improvements
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Predictions
y_pred = model.predict(X_test_tfidf)

# 6. Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Log experiment in MLflow
with mlflow.start_run():
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("ngram_range", "1,2")
    mlflow.log_param("model", "MultinomialNB")

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, pos_label="spam"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, pos_label="spam"))
    mlflow.log_metric("f1", f1_score(y_test, y_pred, pos_label="spam"))

    mlflow.sklearn.log_model(model, "spam_model_nb")
