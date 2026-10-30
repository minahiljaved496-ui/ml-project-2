import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="HR Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'   # encoders, trained models, scores

# ---------- Load encoders ----------
le_dept = joblib.load(MODELS_DIR / 'le_dept.pkl')
le_overtime = joblib.load(MODELS_DIR / 'le_overtime.pkl')
le_attrition = joblib.load(MODELS_DIR / 'le_attrition.pkl')

# ---------- Model registry ----------
# key (sent by frontend) -> (display name, pickle file)
# Naive Bayes is classification-only, so it has no regression entry.
REGRESSION_MODELS = {
    "linear": ("Linear Regression", "linear_regression_model.pkl"),
    "decision_tree": ("Decision Tree", "reg_decision_tree.pkl"),
    "random_forest": ("Random Forest", "reg_random_forest.pkl"),
    "svm": ("SVM (SVR)", "reg_svm.pkl"),
}

CLASSIFICATION_MODELS = {
    "linear": ("Logistic Regression", "logistic_regression_model.pkl"),
    "decision_tree": ("Decision Tree", "clf_decision_tree.pkl"),
    "naive_bayes": ("Naive Bayes", "clf_naive_bayes.pkl"),
    "random_forest": ("Random Forest", "clf_random_forest.pkl"),
    "svm": ("SVM (SVC)", "clf_svm.pkl"),
}

reg_models = {k: joblib.load(MODELS_DIR / f) for k, (_, f) in REGRESSION_MODELS.items()}
clf_models = {k: joblib.load(MODELS_DIR / f) for k, (_, f) in CLASSIFICATION_MODELS.items()}

# Test-set scores saved by train.ipynb (optional, used for display only)
scores_path = MODELS_DIR / 'model_scores.json'
SCORES = json.loads(scores_path.read_text()) if scores_path.exists() else {}


class EmployeeData(BaseModel):
    age: int
    daily_rate: int
    department: str
    distance_from_home: int
    education: int
    environment_satisfaction: int
    job_level: int
    job_satisfaction: int
    monthly_rate: int
    num_companies_worked: int
    over_time: str
    stock_option_level: int
    total_working_years: int
    training_times_last_year: int
    work_life_balance: int
    years_at_company: int


def process_input(data: EmployeeData):
    dept_encoded = le_dept.transform([data.department])[0]
    overtime_encoded = le_overtime.transform([data.over_time])[0]

    features = np.array([[
        data.age, data.daily_rate, dept_encoded, data.distance_from_home,
        data.education, data.environment_satisfaction, data.job_level,
        data.job_satisfaction, data.monthly_rate, data.num_companies_worked,
        overtime_encoded, data.stock_option_level, data.total_working_years,
        data.training_times_last_year, data.work_life_balance, data.years_at_company
    ]])
    return features


def get_model(registry: dict, loaded: dict, model: str, task: str):
    if model not in registry:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not available for {task}. "
                   f"Available: {', '.join(registry.keys())}",
        )
    return registry[model][0], loaded[model]


@app.get("/models")
def list_models():
    """Used by the frontend to build the model menu."""
    reg_scores = SCORES.get("regression_r2", {})
    clf_scores = SCORES.get("classification_accuracy", {})
    score_key = {"linear": ("linear_regression", "logistic_regression")}

    def build(registry, scores, idx):
        out = []
        for key, (name, _) in registry.items():
            skey = score_key[key][idx] if key in score_key else key
            out.append({"key": key, "name": name, "score": scores.get(skey)})
        return out

    return {
        "regression": build(REGRESSION_MODELS, reg_scores, 0),
        "classification": build(CLASSIFICATION_MODELS, clf_scores, 1),
    }


@app.post("/predict/regression")
def predict_salary(data: EmployeeData, model: str = "linear"):
    name, mdl = get_model(REGRESSION_MODELS, reg_models, model, "salary prediction")
    features = process_input(data)
    prediction = mdl.predict(features)
    return {
        "model": name,
        "predicted_monthly_income": round(float(prediction[0]), 2),
    }


@app.post("/predict/classification")
def predict_attrition(data: EmployeeData, model: str = "linear"):
    name, mdl = get_model(CLASSIFICATION_MODELS, clf_models, model, "attrition prediction")
    features = process_input(data)
    prediction = mdl.predict(features)

    status = "High Risk (May Leave)" if prediction[0] == 1 else "Low Risk (Will Stay)"
    return {"model": name, "attrition_risk": status}
