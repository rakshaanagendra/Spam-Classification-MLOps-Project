from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os, json

app = FastAPI(
    title="Spam Classification API",
    description="""
    A simple API that classifies SMS messages as **Spam** or **Ham**  
    using a trained Logistic Regression model with TF-IDF vectorization.  
    The classification threshold is chosen automatically via threshold tuning.
    """,
    version="1.2.0"
)

MODEL_PATH = "outputs/model.pkl"
VECTORIZER_PATH = "outputs/vectorizer.pkl"
THRESHOLD_PATH = "outputs/threshold.json"

model, vectorizer, threshold = None, None, 0.5  # defaults


# -------------------------
# Helper: Load model artifacts
# -------------------------
def load_artifacts():
    global model, vectorizer, threshold
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, "r") as f:
                threshold = json.load(f)["threshold"]
        print(f"✅ Model, vectorizer, and threshold ({threshold}) loaded")
    else:
        model, vectorizer, threshold = None, None, 0.5
        print("⚠️ Model/vectorizer not found. Run training first.")


# Load once at startup
load_artifacts()


class Message(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {"text": "Win a free iPhone now!!!"}
        }


@app.get("/health", summary="Health check")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "threshold": threshold,
    }


@app.post("/predict", summary="Classify a message")
def predict(msg: Message):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model or vectorizer not loaded")

    X = vectorizer.transform([msg.text])
    prob_spam = model.predict_proba(X)[0][list(model.classes_).index("spam")]
    prediction = "spam" if prob_spam >= threshold else "ham"

    return {
        "text": msg.text,
        "prediction": prediction,
        "prob_spam": round(float(prob_spam), 3),
        "threshold": threshold,
    }


@app.post("/reload-model", summary="Reload model artifacts")
def reload_model():
    load_artifacts()
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Artifacts not found. Train first.")
    return {"status": "reloaded", "threshold": threshold}
