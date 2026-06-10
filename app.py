import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="Nowcast Inondation", page_icon="💦", layout="wide")

# ─── SECRETS ───────────────────────────────────────────────────────────
# Utilise st.secrets["HUBEAU_API_URL"] que tu configureras dans l'interface Hugging Face
HUBEAU_API = st.secrets.get("HUBEAU_API_URL", "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr")

# ─── STATIONS ──────────────────────────────────────────────────────────
STATIONS = {
    "A161003001": {"nom": "Ill à Colmar", "lat": 48.080, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 800, "seuil_max": 2500},
    "A214010001": {"nom": "Fecht à Ostheim", "lat": 48.166, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 300, "seuil_max": 2200},
    "A236003001": {"nom": "Ill à Kogenheim", "lat": 48.266, "lon": 7.533, "seuil_alerte": 2000, "seuil_min": 400, "seuil_max": 2000},
    "A348020001": {"nom": "Zorn à Waltenheim", "lat": 48.716, "lon": 7.583, "seuil_alerte": 2000, "seuil_min": 500, "seuil_max": 1800},
}

# ─── CHARGEMENT MODÈLE ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

model = load_model()

# ─── CHARGEMENT DONNÉES ────────────────────────────────────────────────
@st.cache_data
def load_local_data():
    dfs = {}
    path = "hubeau"
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".csv"):
                code = f.split("_")[2]
                df = pd.read_csv(os.path.join(path, f))
                df.columns = df.columns.str.lower().str.strip()
                df["date_obs"] = pd.to_datetime(df["date_obs"])
                dfs[code] = df
    return dfs

dfs_local = load_local_data()

# ─── UI ────────────────────────────────────────────────────────────────
st.title("💦 Nowcast Inondation — Bassin de l'Ill")

# Paramètres
col_left, col_right = st.columns([1, 3])
with col_left:
    selected_nom = st.selectbox("Choisir une station", [info["nom"] for info in STATIONS.values()])
    selected_code = [c for c, i in STATIONS.items() if i["nom"] == selected_nom][0]
    info = STATIONS[selected_code]
    
    date_debut = st.date_input("Date début", value=datetime(2020, 1, 1))
    seuil_alerte = st.number_input("Seuil alerte (mm)", value=info["seuil_alerte"])

with col_right:
    st.subheader(f"📈 Historique — {selected_nom}")
    df = dfs_local.get(selected_code)
    if df is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date_obs"], y=df["resultat_obs"], name="Obs", line=dict(color='grey')))
        fig.add_hline(y=seuil_alerte, line_color="red", line_dash="dash")
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

# ─── SIMULATION ────────────────────────────────────────────────────────
st.subheader("🔮 Simulateur")
h = st.slider("Hauteur (mm)", info["seuil_min"], info["seuil_max"])
if model:
    pred = int(model.predict(pd.DataFrame([[h, 12, 6]], columns=["resultat_obs", "hour", "month"]))[0])
    st.info(f"Prédiction risque (1-3) : {pred}")