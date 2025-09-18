# 🚀 Spam Classifier Project — Commands Cheat Sheet

This file contains all the important commands for training, tracking, Docker deployment, and testing.

---

## 🔄 DVC & MLflow

```powershell
# Start MLflow UI (open http://127.0.0.1:5000 in your browser)
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

# Reproduce DVC pipeline (runs stages if code/data changed)
dvc repro


## 🐳 Build Docker image
docker build -t spam-classifier:latest .

# Stop and remove old container (ignore errors if not running)
docker stop spam-api
docker rm spam-api

# Run new container (maps port 8000 and mounts outputs/)
docker run -d --restart unless-stopped -p 8000:8000 ` -v C:\Users\nagen\Desktop\Kaggle_datasets\SMS-Spam-collection\Spam-Classification-MLOps-Project\outputs:/app/outputs ` --name spam-api spam-classifier:latest


## ⚡ Reload model
# Normally, FastAPI app loads the trained model when the container starts.
#Docker starts → main.py loads model.pkl, vectorizer.pkl, threshold.json.
#If you retrain later and produce new versions of those files → the running container still uses the old model (until you restart Docker).
#A /reload-model endpoint fixes that:
#It tells the API to re-read the files from outputs/ while the container is still running.
#No need to stop/start Docker.

Invoke-RestMethod -Method POST -Uri "http://localhost:8000/reload-model"


## ✅ Health Check - Confirm API is running and see if model/vectorizer/threshold are loaded

Invoke-RestMethod -Method GET -Uri "http://localhost:8000/health"


## 📊 Prediction Test

Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict" ` -ContentType "application/json" ` -Body '{"text":"Win a free vacation prize now!!!"}'


## 🌐 Swagger UI & ReDoc

- Swagger UI (interactive): http://localhost:8000/docs
- ReDoc (documentation view): http://localhost:8000/redoc
