import pandas as pd                                 # Data handling
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Function to create new features
def add_text_features(df, text_col="message"):
    s = df[text_col].astype(str)  # ensure text
    df["msg_len"] = s.str.len()   # total characters
    df["digit_count"] = s.str.count(r"\d")  # count digits
    df["punct_count"] = s.str.count(r"[!?.:,;]")  # count punctuation
    num_upper = s.str.count(r"[A-Z]")  # uppercase letters
    df["upper_ratio"] = num_upper / df["msg_len"].replace(0, 1)  # avoid div/0
    return df

# 2. Load dataset
data = pd.read_csv(r"C:\Users\nagen\Desktop\Kaggle_datasets\SMS Spam collection\Spam-Classification-MLOps-Project\data\smsspamcollection\SMSSpamCollection",
                   sep="\t", header=None, names=["label", "message"])


# Add engineered features
data = add_text_features(data, "message")

# 3. Split into X (features) and y (target)
X = data[["message", "msg_len", "digit_count", "punct_count", "upper_ratio"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocessing with ColumnTransformer
text_transformer = TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1,2))
preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", text_transformer, "message"),
        ("num", StandardScaler(), ["msg_len", "digit_count", "punct_count", "upper_ratio"])
    ],
    remainder="drop"
)

# 5. Pipeline: Preprocessor → Classifier
clf = LogisticRegression(max_iter=1000, C=10, class_weight="balanced")
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", clf)
])

# 6. Train
pipeline.fit(X_train, y_train)

# 7. Predict
y_pred = pipeline.predict(X_test)

# 8. Evaluate
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (spam):", precision_score(y_test, y_pred, pos_label="spam"))
print("Recall (spam):", recall_score(y_test, y_pred, pos_label="spam"))
print("F1 (spam):", f1_score(y_test, y_pred, pos_label="spam"))
