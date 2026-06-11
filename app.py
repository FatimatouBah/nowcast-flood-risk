import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="💦 Nowcast Crues", page_icon="💦", layout="wide")

API_BASE_URL = st.secrets.get("API_BASE_URL", "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space")

# ====================== CHARGEMENT ======================
@st.cache_data(ttl=3600)
def load_stations():
    try:
        r = requests.get(f"{API_BASE_URL}/stations", timeout=10)
        r.raise_for_status()
        stations = r.json().get("stations", [])
        return stations
    except Exception as e:
        st.error(f"❌ Erreur API : {e}")
        return []

stations = load_stations()
if not stations:
    st.stop()

st.title("💦 Nowcast — Risque d'Inondation")
st.markdown("**Bassin versant de l'Ill - Grand Est**")

# ====================== CARTE (version la plus fiable sur HF) ======================
st.subheader("🗺️ Bassin versant de l'Ill (Grand Est)")

# Conversion en DataFrame pour st.map (plus stable sur HF)
df_map = pd.DataFrame(stations)

st.map(
    df_map,
    latitude="latitude",
    longitude="longitude",
    use_container_width=True,
    size=100,
    color="#003189"
)

# ====================== PARAMÈTRES ======================
st.subheader("⚙️ Paramètres")

col1, col2 = st.columns([3, 2])
with col1:
    station_dict = {s["label"]: s["code"] for s in stations}
    selected_name = st.selectbox("Choisir une station", options=list(station_dict.keys()))
    selected_code = station_dict[selected_name]

with col2:
    alert_threshold_m = st.number_input("Seuil alerte (m)", value=2.0, min_value=0.5, max_value=10.0, step=0.1)

col_d1, col_d2 = st.columns(2)
with col_d1:
    start_date = st.date_input("Date début", value=datetime(2025, 6, 1))
with col_d2:
    end_date = st.date_input("Date fin", value=datetime.today().date() + timedelta(days=30))

# ====================== GRAPH (simplifié) ======================
st.subheader(f"📈 Hauteur d'eau — {selected_name}")

st.info("🔄 Les données d'observations et prédictions sont en cours de chargement...")

# On affiche au moins un message clair si pas de données
st.caption("Données : Hub'eau API — Grand Est • Version optimisée HF")

# Pour le moment on laisse un espace pour le graphique futur
st.markdown("---")
st.success("L'application est bien déployée sur Hugging Face ! 🎉\n\nLes graphiques seront ajoutés dès que l'API renverra des données.")