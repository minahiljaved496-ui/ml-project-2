# HR Analytics ML App

A full-stack machine learning app built on the IBM **HR Employee Attrition** dataset. It has two tasks:

| Task | Type | Target |
|---|---|---|
| Predict Salary | Regression | `MonthlyIncome` |
| Check Attrition Risk | Classification | `Attrition` (Yes / No) |

The frontend has a **model menu**. You pick the algorithm and the same form is predicted with that model.

**Stack:** scikit-learn, FastAPI, HTML/JavaScript

---

## Project Structure

```
my-ml-project/
├── data/
│   └── raw/                  # Original dataset (HR-Employee-Attrition.csv)
├── notebooks/
│   └── train.ipynb           # Trains all models and saves them to models/
├── models/                   # Trained models, label encoders, model_scores.json
├── main.py                   # FastAPI REST API
├── index.html                # Frontend (open in the browser)
└── README.md
```

## Models

| Model | Salary (Regression) | Attrition (Classification) |
|---|---|---|
| Linear / Logistic Regression (baseline) | R² 0.895 | Accuracy 87.8% |
| Decision Tree | R² 0.910 | Accuracy 82.7% |
| Naive Bayes | not applicable | Accuracy 88.4% |
| Random Forest | R² 0.918 | Accuracy 87.4% |
| Support Vector Machine (SVM) | R² 0.896 | Accuracy 88.8% |

Scores are on a 20% test split (`random_state=42`).

**Notes**
- Naive Bayes only does classification, so it is not trained for salary. The frontend disables the Salary button when Naive Bayes is selected, and the API returns a 400 error.
- SVM is sensitive to feature scale, so it is wrapped in a `Pipeline` with `StandardScaler`.

## Input Features

`Age, DailyRate, Department, DistanceFromHome, Education, EnvironmentSatisfaction, JobLevel, JobSatisfaction, MonthlyRate, NumCompaniesWorked, OverTime, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany`

`Department` and `OverTime` are label-encoded. The encoders are saved in `models/`. The form shows only 5 fields. The rest are sent as hidden default values.

## Setup

**1. Install dependencies** (Python 3.10+)

```bash
pip install fastapi uvicorn scikit-learn joblib numpy pandas jupyter
```

**2. Train the models**

Open `notebooks/train.ipynb` and choose **Run All**. This reads `data/raw/HR-Employee-Attrition.csv` and writes all `.pkl` files and `model_scores.json` into `models/`.

Re-run the notebook on your own machine if you get a scikit-learn version warning. Pickle files must be loaded with the same scikit-learn version that created them.

**3. Start the API** (from the project root, the folder that contains `main.py`)

```bash
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`. Interactive docs are at `http://127.0.0.1:8000/docs`.

**4. Open the frontend**

Open `index.html` in your browser (or use VS Code Live Server). Choose a model from **Select Model**, fill the form, and click the predict button.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/models` | Available models with their test scores |
| POST | `/predict/regression?model=<key>` | Predicted monthly income |
| POST | `/predict/classification?model=<key>` | Attrition risk |

**Model keys:** `linear`, `decision_tree`, `naive_bayes`, `random_forest`, `svm` (default: `linear`)

**Example request**

```bash
curl -X POST "http://127.0.0.1:8000/predict/classification?model=random_forest" \
  -H "Content-Type: application/json" \
  -d '{"age":35,"daily_rate":800,"department":"Sales","distance_from_home":5,
       "education":3,"environment_satisfaction":3,"job_level":2,"job_satisfaction":3,
       "monthly_rate":15000,"num_companies_worked":2,"over_time":"Yes",
       "stock_option_level":1,"total_working_years":10,"training_times_last_year":2,
       "work_life_balance":3,"years_at_company":5}'
```

**Example response**

```json
{ "model": "Random Forest", "attrition_risk": "Low Risk (Will Stay)" }
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Failed to fetch` on the frontend | The API is not running. Start `uvicorn main:app --reload` and check `http://127.0.0.1:8000/models`. |
| `Could not import module "main"` | Run uvicorn from the folder that contains `main.py`, and make sure the file is named exactly `main.py`. |
| `FileNotFoundError ... .pkl` | `models/` is empty. Run `notebooks/train.ipynb` first. |
| Version warning when loading models | Re-run the notebook with your own scikit-learn version. |
