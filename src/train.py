# 1. Import libraries
import pandas as pd                              # For data handling (reading CSV, DataFrame operations)
from sklearn.model_selection import train_test_split   # To split data into training & testing
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text into numeric features
from sklearn.linear_model import LogisticRegression           # Our first ML model
from sklearn.metrics import classification_report, confusion_matrix  # Metrics for evaluation
import mlflow                                # MLflow for experiment tracking
import mlflow.sklearn                        # MLflow helper for sklearn models

# 2. Load dataset
data = pd.read_csv(r"C:\Users\nagen\Desktop\Kaggle_datasets\SMS Spam collection\Spam-Classification-MLOps-Project\data\smsspamcollection\SMSSpamCollection",
                   sep="\t", header=None, names=["label", "message"])

# -  read the dataset as a DataFrame.
# - `sep="\t"` means "split by tab character" since file is tab-separated.
# - `header=None` because the file has no column names in the first row.
# - `names=["label", "message"]` gives our columns proper names.

# 3. Split dataset into features (X) and labels (y)
X = data['message']    # All the text messages
y = data['label']      # 'ham' or 'spam'

# 4. Train/Test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# - test_size=0.2 → 20% test data, 80% training.
# - random_state=42 → makes results reproducible.
# - stratify=y → ensures spam/ham ratio is the same in train and test.

# 5. Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# - TF-IDF = Term Frequency - Inverse Document Frequency, measures how important a word is.
# - stop_words="english" → removes common words like "the", "and", "is".
# - max_features=5000 → keep only the 5000 most frequent words.

# 6. Define and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
# - max_iter=1000 → ensures convergence (default 100 may not be enough).
# - .fit() trains the model using TF-IDF features and labels.

# 7. Make predictions
y_pred = model.predict(X_test_tfidf)

# 8. Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# - classification_report → precision, recall, f1-score for each class (ham/spam).
# - confusion_matrix → shows counts of true positives/false positives etc.

# 9. Log experiment with MLflow
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("model", "LogisticRegression")

    # Log metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, pos_label="spam"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, pos_label="spam"))
    mlflow.log_metric("f1", f1_score(y_test, y_pred, pos_label="spam"))

    # Log model
    mlflow.sklearn.log_model(model, "spam_model")
