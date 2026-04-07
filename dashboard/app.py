import streamlit as st
import pandas as pd
import numpy as np
import shap
import mlflow.sklearn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='RecidivAI — Transparent Recidivism Risk',
    page_icon='⚖️',
    layout='wide'
)

RUN_ID = 'ffc2fefec05940c7abdf92dda52b8360'

@st.cache_resource
def load_model_and_scaler():
    model = mlflow.sklearn.load_model(f'runs:/{RUN_ID}/model')
    scaler = mlflow.sklearn.load_model(f'runs:/{RUN_ID}/scaler')
    return model, scaler

model, scaler = load_model_and_scaler()

FEATURE_COLS = [
    'age', 'priors_count14', 'juv_fel_count', 'juv_misd_count',
    'is_juvenile_offender', 'prior_crime_density', 'high_prior_count',
    'charge_severity_score', 'sex_binary'
]

st.title('⚖️ RecidivAI — Violent Recidivism Risk Assessment')
st.caption(
    'A transparent, explainable alternative to COMPAS. Built on ProPublica COMPAS data. '
    'For educational and research purposes only.'
)

left_col, right_col = st.columns([1, 1.5])

with left_col:
    st.subheader('Defendant Features')

    age = st.slider('Age', min_value=18, max_value=70, value=30)
    priors_count14 = st.number_input('Number of Prior Charges', min_value=0, max_value=50, value=0)
    juv_fel = st.number_input('Juvenile Felony Charges', min_value=0, max_value=20, value=0)
    juv_mis = st.number_input('Juvenile Misdemeanor Charges', min_value=0, max_value=20, value=0)
    charge = st.selectbox('Current Charge Degree', ['Felony', 'Misdemeanor'])
    sex = st.selectbox('Sex', ['Male', 'Female'])

    is_juv = 1 if (juv_fel + juv_mis) > 0 else 0
    density = priors_count14 / max(age - 18, 1)
    high_prior = 1 if priors_count14 > 3 else 0
    charge_score = 2 if charge == 'Felony' else 1
    sex_bin = 1 if sex == 'Male' else 0

    if st.button('Run Risk Assessment', type='primary'):
        features_raw = np.array([[
            age, priors_count14, juv_fel, juv_mis,
            is_juv, density, high_prior, charge_score, sex_bin
        ]])

        features_scaled = scaler.transform(features_raw)

        risk_prob = model.predict_proba(features_scaled)[0][1]
        prediction = int(risk_prob >= 0.5)

        with right_col:
            st.subheader('Risk Assessment Results')

            if prediction == 1:
                st.error(f'HIGH RISK | Score: {risk_prob:.1%}')
            else:
                st.success(f'LOW RISK | Score: {risk_prob:.1%}')

            st.metric('Risk Score', f'{risk_prob:.1%}')
            st.progress(float(risk_prob))

            st.subheader('Why This Prediction? (SHAP Explanation)')

            explainer = shap.LinearExplainer(
                model,
                shap.maskers.Independent(features_scaled)
            )
            shap_vals = explainer.shap_values(features_scaled)

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    data=features_raw[0],
                    feature_names=FEATURE_COLS
                ),
                show=False
            )
            st.pyplot(fig)