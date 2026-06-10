import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nowcast Inondation — Ill Grand Est", page_icon="💦", layout="wide")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    h1, h2, h3 { color: #003189; }
</style>
""", unsafe_allow_html=True)

# ─── SECRETS (Configuration sur Hugging Face : Settings > Variables and secrets) ───
HUBEAU_API = st.secrets.get("HUBEAU_API_URL", "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr")

# ─── STATIONS ──────────────────────────────────────────────────────────────────
STATIONS = {
    "A161003001": {"nom": "Ill à Colmar",      "lat": 48.080, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 800,  "seuil_max": 2500},
    "A214010001": {"nom": "Fecht à Ostheim",   "lat": 48.166, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 300,  "seuil_max": 2200},
    "A236003001": {"nom": "Ill à Kogenheim",   "lat": 48.266, "lon": 7.533, "seuil_alerte": 2000, "seuil_min": 400,  "seuil_max": 2000},
    "A348020001": {"nom": "Zorn à Waltenheim", "lat": 48.716, "lon": 7.583, "seuil_alerte": 2000, "seuil_min": 500,  "seuil_max": 1800},
}

# ─── CHARGEMENT ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"): return joblib.load("model.pkl")
    return None

@st.cache_data
def load_local_data():
    dfs = {}
    path = "hubeau"
    if os.path.exists(path):
        for f in [x for x in os.listdir(path) if x.endswith(".csv")]:
            code = f.split("_")[2]
            df = pd.read_csv(os.path.join(path, f))
            df.columns = df.columns.str.lower().str.strip()
            df["date_obs"] = pd.to_datetime(df["date_obs"])
            dfs[code] = df.sort_values("date_obs")
    return dfs

model = load_model()
dfs = load_local_data() # Appel corrigé ici

# ─── INTERFACE ────────────────────────────────────────────────────────────────
st.title("💦 Nowcast — Risque d'Inondation")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ Paramètres")
    selected_nom = st.selectbox("Choisir une station", [info["nom"] for info in STATIONS.values()])
    selected_code = [c for c, i in STATIONS.items() if i["nom"] == selected_nom][0]
    
    date_debut = st.date_input("Date début", value=datetime(2020, 1, 1))
    date_fin = st.date_input("Date fin", value=datetime.now())
    
    st.markdown("---")
    st.subheader("⚠️ Ajuster seuils")
    seuil_alerte = st.number_input("Seuil alerte (mm)", value=STATIONS[selected_code]["seuil_alerte"])
    seuil_min = st.number_input("Min (mm)", value=STATIONS[selected_code]["seuil_min"])
    seuil_max = st.number_input("Max (mm)", value=STATIONS[selected_code]["seuil_max"])

with col_right:
    st.subheader(f"📈 Historique — {selected_nom}")
    df = dfs.get(selected_code)
    
    if df is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date_obs"], y=df["resultat_obs"], line=dict(color='grey')))
        fig.add_hline(y=seuil_alerte, line_color="red", line_dash="dash")
        
        fig.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            yaxis=dict(gridcolor="#eeeeee")
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── PRÉDICTION ────────────────────────────────────────────────────────────────
st.subheader("🔮 Simulation")
h_sim = st.slider("Hauteur d'eau simulée (mm)", seuil_min, seuil_max)

if model:
    pred = int(model.predict(pd.DataFrame([[h_sim, 12, 6]], columns=["resultat_obs", "hour", "month"]))[0])
    risk = {1: ("🟢 Faible", "#d4edda"), 2: ("🟡 Modéré", "#fff3cd"), 3: ("🔴 Élevé", "#f8d7da")}
    lbl, color = risk.get(pred, ("Inconnu", "#eee"))
    st.markdown(f"<div style='background:{color}; padding:20px; border-radius:10px; text-align:center'>Risque : {lbl}</div>", unsafe_allow_html=True)

st.caption("Fatimatou Bah — Master IA 2026")