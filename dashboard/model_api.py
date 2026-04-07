from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np

app = FastAPI(title='RecidivAI Risk API', version='1.0')

RUN_ID = 'ffc2fefec05940c7abdf92dda52b8360'
BASE_URI = f'file:../mlruns'

class DefendantFeatures(BaseModel):
    age: int
    priors_count14: int  # was priors_count
    juv_fel_count: int
    juv_misd_count: int
    is_juvenile_offender: int
    prior_crime_density: float
    high_prior_count: int
    charge_severity_score: int
    sex_binary: int

@app.get('/')
def root():
    return {'status': 'RecidivAI API is running'}

@app.post('/predict')
def predict(features: DefendantFeatures):
    model = mlflow.sklearn.load_model(f'runs:/{RUN_ID}/model')
    scaler = mlflow.sklearn.load_model(f'runs:/{RUN_ID}/scaler')

    feature_array = np.array([[
        features.age,
        features.priors_count14,  # was features.priors_count,
        features.juv_fel_count,
        features.juv_misd_count,
        features.is_juvenile_offender,
        features.prior_crime_density,
        features.high_prior_count,
        features.charge_severity_score,
        features.sex_binary,
    ]])

    feature_array_scaled = scaler.transform(feature_array)

    risk_score = float(model.predict_proba(feature_array_scaled)[0][1])
    prediction = int(risk_score >= 0.5)

    return {
        'risk_score': round(risk_score, 4),
        'prediction': prediction,
        'risk_label': 'High Risk' if prediction == 1 else 'Low Risk'
    }
