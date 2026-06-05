import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import mlflow.sklearn

st.set_page_config(page_title="Nowcast Flood Risk", page_icon="🌊")

st.title("🌊 Nowcast Risque d'Inondation — Bassin de l'Ill")
st.markdown("Prédiction du niveau de risque en quasi-temps réel à partir des données Hub'eau.")

@st.cache_resource
def load_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["1"], order_by=["metrics.f1 DESC"], max_results=1)
    run_id = runs[0].info.run_id
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    return model

model = load_model()

st.sidebar.header("Paramètres de la station")
hauteur = st.sidebar.slider("Hauteur d'eau (mm)", 400, 1500, 800)
heure = st.sidebar.slider("Heure", 0, 23, 12)
mois = st.sidebar.slider("Mois", 1, 12, 6)

X = pd.DataFrame([[hauteur, heure, mois]], columns=["resultat_obs", "hour", "month"])
prediction = model.predict(X)[0]

risk_labels = {1: "🟢 Faible", 2: "🟡 Modéré", 3: "🔴 Élevé"}
risk_colors = {1: "green", 2: "orange", 3: "red"}

st.markdown("---")
st.subheader("Résultat de la prédiction")
st.markdown(f"<h1 style='color:{risk_colors[prediction]}'>{risk_labels[prediction]}</h1>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Données saisies")
st.dataframe(X)