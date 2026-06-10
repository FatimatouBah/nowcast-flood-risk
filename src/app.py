import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="Nowcast Inondation — Ill Grand Est", page_icon="💦", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #003189; }
</style>
""", unsafe_allow_html=True)

# ─── SECRETS ───────────────────────────────────────────────────────────────────
HUBEAU_API = st.secrets.get("HUBEAU_API_URL", "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr")

# ─── STATIONS ──────────────────────────────────────────────────────────────────
STATIONS = {
    "A161003001": {"nom": "Ill à Colmar",      "lat": 48.080, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 800,  "seuil_max": 2500},
    "A214010001": {"nom": "Fecht à Ostheim",   "lat": 48.166, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 300,  "seuil_max": 2200},
    "A236003001": {"nom": "Ill à Kogenheim",   "lat": 48.266, "lon": 7.533, "seuil_alerte": 2000, "seuil_min": 400,  "seuil_max": 2000},
    "A348020001": {"nom": "Zorn à Waltenheim", "lat": 48.716, "lon": 7.583, "seuil_alerte": 2000, "seuil_min": 500,  "seuil_max": 1800},
}

# ─── CHARGEMENT MODÈLE ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

# ─── CHARGEMENT DONNÉES LOCALES ────────────────────────────────────────────────
@st.cache_data
def load_local_data():
    dfs = {}
    path = "hubeau"
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith(".csv")]
        for f in sorted(files):
            code = f.split("_")[2]
            df = pd.read_csv(os.path.join(path, f))
            df.columns = df.columns.str.lower().str.strip()
            df["date_obs"] = pd.to_datetime(df["date_obs"])
            dfs[code] = df.sort_values("date_obs")
    return dfs

# ─── CHARGEMENT HISTORIQUE ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_hubeau_history(code_station, date_debut, date_fin):
    try:
        url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
        params = {
            "code_entite": code_station, "grandeur_hydro_elab": "QmnJ",
            "date_debut_obs_elab": date_debut.strftime("%Y-%m-%d"),
            "date_fin_obs_elab": date_fin.strftime("%Y-%m-%d"),
            "size": 2000, "fields": "date_obs_elab,resultat_obs_elab"
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code in [200, 206]:
            data = r.json().get("data", [])
            df = pd.DataFrame(data)
            df["date_obs_elab"] = pd.to_datetime(df["date_obs_elab"])
            return df.sort_values("date_obs_elab")
    except: return None
    return None

# ─── PRÉDICTION ────────────────────────────────────────────────────────────────
def get_risk(hauteur, heure, mois, model):
    if model is None or hauteur == 0: return 1
    X = pd.DataFrame([[hauteur, heure, mois]], columns=["resultat_obs", "hour", "month"])
    return max(1, min(int(model.predict(X)[0]), 3))

model = load_model()
dfs_local = load_local_data() # <--- CORRIGÉ : Appel de la fonction définie

# ─── INTERFACE ────────────────────────────────────────────────────────────────
st.title("💦 Nowcast — Risque d'Inondation")
# ... (le reste de ton code d'affichage ne change pas)