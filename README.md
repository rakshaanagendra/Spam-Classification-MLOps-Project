# Spam Classification with MLOps

This project classifies SMS messages as **spam** or **ham** using Logistic Regression and feature engineering.  
It demonstrates an **MLOps workflow** with:
- MLflow for experiment tracking,
- DVC for reproducible pipelines,
- Git for version control.

## Project Structure

├── data/                     # Dataset (not pushed, tracked with DVC)
├── outputs/                  # Model outputs (metrics, artifacts)
├── src/
│   └── train_log_mlflow.py   # Training + MLflow logging script
├── dvc.yaml                  # DVC pipeline definition
├── dvc.lock                  # Pipeline lock file
├── requirements.txt          # Python dependencies
└── README.md

## Results

### Confusion Matrix (Best Threshold)
The confusion matrix below shows the model’s predictions at the best threshold identified (optimized for F1 score):

![Confusion Matrix](assets/conf_matrix.png)

---

### Threshold vs F1 Curve
This plot shows how the F1 score varies with the decision threshold. The highlighted best threshold maximizes performance:

![Threshold vs F1](assets/threshold_f1.png)


## Setup
```bash
# Clone repo
git clone https://github.com/your-username/Spam-Classification-MLOps-Project.git
cd Spam-Classification-MLOps-Project

# Create environment
python -m venv .venv
.venv\Scripts\activate  # (Windows)

pip install -r requirements.txt

# Reproduce pipeline
dvc repro

# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db


