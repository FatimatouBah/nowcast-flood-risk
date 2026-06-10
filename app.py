import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# 1. Configuration
st.set_page_config(page_title="Nowcast Inondation", page_icon="💦", layout="wide")

# 2. Gestion API (Sécurisée)
try:
    HUBEAU_API = st.secrets.get("HUBEAU_API_URL", os.getenv("HUBEAU_API_URL", "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr"))
except:
    HUBEAU_API = os.getenv("HUBEAU_API_URL", "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr")

# 3. Stations
STATIONS = {
    "A161003001": {"nom": "Ill à Colmar",      "lat": 48.080, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 800,  "seuil_max": 2500},
    "A214010001": {"nom": "Fecht à Ostheim",   "lat": 48.166, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 300,  "seuil_max": 2200},
    "A236003001": {"nom": "Ill à Kogenheim",   "lat": 48.266, "lon": 7.533, "seuil_alerte": 2000, "seuil_min": 400,  "seuil_max": 2000},
    "A348020001": {"nom": "Zorn à Waltenheim", "lat": 48.716, "lon": 7.583, "seuil_alerte": 2000, "seuil_min": 500,  "seuil_max": 1800},
}

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"): return joblib.load("model.pkl")
    return None

@st.cache_data
def load_local_data():
    dfs = {}
    path = "hubeau"
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".csv"):
                try:
                    code = f.split("_")[2]
                    df = pd.read_csv(os.path.join(path, f))
                    df.columns = df.columns.str.lower().str.strip()
                    if "date_obs" in df.columns and "resultat_obs" in df.columns:
                        df["date_obs"] = pd.to_datetime(df["date_obs"])
                        dfs[code] = df.sort_values("date_obs")
                except: continue
    return dfs

def get_risk(hauteur, heure, mois, model):
    if model is None or hauteur == 0: return 1
    try:
        X = pd.DataFrame([[hauteur, heure, mois]], columns=model.feature_names_in_)
        return max(1, min(int(model.predict(X)[0]), 3))
    except: return 1

risk_labels = {1: "🟢 Faible", 2: "🟡 Modéré", 3: "🔴 Élevé"}
model, dfs_local = load_model(), load_local_data()

# ════════ INTERFACE ════════
st.markdown("# 💦 Nowcast — Risque d'Inondation")
st.caption("Projet réalisé par Fatimatou Bah")
st.markdown("### Bassin versant de l'Ill — Grand Est")
st.markdown("---")

# Carte
st.subheader("🗺️ Stations hydrométriques")
fig_map = go.Figure(go.Scattermapbox(lat=[s["lat"] for s in STATIONS.values()], lon=[s["lon"] for s in STATIONS.values()], 
                                     mode='markers+text', marker=dict(size=25, color="red")))
fig_map.update_layout(mapbox=dict(style="carto-positron", center=dict(lat=48.25, lon=7.45), zoom=9.5), 
                      margin=dict(l=0, r=0, t=0, b=0), height=450)
st.plotly_chart(fig_map, use_container_width=True)

# Paramètres
col_left, col_right = st.columns([1, 2])
with col_left:
    st.subheader("⚙️ Paramètres")
    selected_nom = st.selectbox("Choisir une station", [i["nom"] for i in STATIONS.values()])
    selected_code = [c for c, i in STATIONS.items() if i["nom"] == selected_nom][0]
    info = STATIONS[selected_code]
    date_debut = st.date_input("Date début", value=datetime(2020, 1, 1), min_value=datetime(2007, 1, 1))
    date_fin = st.date_input("Date fin", value=datetime.now())
    seuil_alerte = st.number_input("Seuil d'alerte (mm)", value=2000)

with col_right:
    st.subheader(f"📈 Analyse — {selected_nom}")
    df_local = dfs_local.get(selected_code)
    if df_local is not None:
        df_clean = df_local.copy()
        if df_clean["date_obs"].dt.tz is not None: df_clean["date_obs"] = df_clean["date_obs"].dt.tz_localize(None)
        df_f = df_clean[(df_clean["date_obs"] >= pd.Timestamp(date_debut)) & (df_clean["date_obs"] <= pd.Timestamp(date_fin))]
        
        if not df_f.empty:
            st.metric("Hauteur maximale observée", f"{df_f['resultat_obs'].max():.0f} mm")
            st.write(f"Du {df_f['date_obs'].min().strftime('%d/%m/%Y')} au {df_f['date_obs'].max().strftime('%d/%m/%Y')}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_f["date_obs"], y=df_f["resultat_obs"], line=dict(color='gray')))
            fig.add_hline(y=seuil_alerte, line_dash="dash", line_color="red")
            fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)), plot_bgcolor="white", height=400)
            st.plotly_chart(fig, use_container_width=True)

# Simulateur
st.subheader("🔮 Simulateur de prédiction")
c1, c2, c3 = st.columns(3)
h_sim = c1.slider("Hauteur (mm)", info["seuil_min"], info["seuil_max"], int((info["seuil_min"]+info["seuil_max"])/2))
hr_sim = c2.slider("Heure", 0, 23, datetime.now().hour)
mo_sim = c3.slider("Mois", 1, 12, datetime.now().month)

pred = get_risk(h_sim, hr_sim, mo_sim, model)
st.markdown(f"### Résultat : {risk_labels[pred]}", unsafe_allow_html=True)
st.caption("Données : Hub'eau API — Mastère Architecte IA — Jedha 2026 — Fatimatou Bah")