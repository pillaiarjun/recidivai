# RecidivAI — Transparent Violent Recidivism Risk Prediction

## What This Project Does
A transparent, auditable ML pipeline that predicts violent bail recidivism
using the ProPublica COMPAS dataset, with explicit fairness analysis across
racial groups and SHAP-based explainability for every prediction.

## Why It Was Built
In 2016, ProPublica published an investigation into COMPAS — a black-box
algorithm used by courts in 20+ US states to decide whether defendants should
be released on bail. They found the algorithm was twice as likely to
incorrectly flag Black defendants as future criminals compared to white
defendants with similar histories. This project builds a transparent,
auditable alternative: every prediction comes with a SHAP explanation showing
exactly which features drove the risk score and by how much.

## Architecture
```
ProPublica CSV
       |
       v
[BRONZE LAYER] — Raw data, unchanged, stored in Delta Lake
       |
       v  PySpark ELT
[SILVER LAYER] — Nulls dropped, types enforced, duplicates removed
       |
       v  PySpark feature engineering
[GOLD LAYER]  — 5 engineered features, ML-ready Delta table
       |
       v  Scikit-learn + MLflow
[MODEL REGISTRY] — Best model versioned in MLflow
       |
       v  FastAPI
[REST API] — POST /predict returns risk score
       |
       v  Streamlit
[DASHBOARD] — Interactive UI with SHAP waterfall explanation
```

## Tech Stack
| Tool | Purpose |
|---|---|
| PySpark | ELT transformations across Medallion layers |
| Delta Lake | ACID storage format for Bronze/Silver/Gold |
| Scikit-learn | Model training (Logistic Regression, Random Forest, GBM) |
| MLflow | Experiment tracking and model registry |
| SHAP | Per-prediction explainability via LinearExplainer |
| FastAPI | REST endpoint for model serving |
| Streamlit | Interactive dashboard |

## Results
Best model: Logistic Regression (class_weight=balanced)

| Metric | Score |
|---|---|
| ROC-AUC | 0.7405 |
| F1-Score | 0.4372 |
| Precision (high risk) | 0.35 |
| Recall (high risk) | 0.57 |

### Fairness Analysis — False Positive Rate by Race
| Race | N | FPR (Our Model) | FNR (Our Model) |
|---|---|---|---|
| African-American | 432 | 0.322 | 0.330 |
| Native American | 5 | 0.200 | 0.000 |
| Asian | 8 | 0.200 | 0.333 |
| Caucasian | 353 | 0.160 | 0.634 |
| Hispanic | 96 | 0.112 | 0.375 |
| Other | 55 | 0.044 | 0.600 |

The 2x FPR disparity between African-American and Caucasian defendants
persists even without race as a model feature, driven by correlations between
prior charge counts and race due to differential policing patterns. This
reflects the Chouldechova impossibility result: when base rates differ across
groups, equalizing FPR and FNR simultaneously is mathematically impossible.

## How to Run
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Download dataset:
   `curl -L -o data/raw/compas-scores-two-years-violent.csv 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years-violent.csv'`
4. Run notebooks in order: 01 → 02 → 03 → 04 → 05
5. `streamlit run dashboard/app.py`

## Ethical Note
This project is for educational and research purposes only. The model
should never be used to inform actual bail or sentencing decisions. The
fairness analysis intentionally surfaces the limitations of algorithmic
risk assessment in criminal justice contexts.