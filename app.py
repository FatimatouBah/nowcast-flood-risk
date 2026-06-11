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
        return r.json().get("stations", [])
    except Exception as e:
        st.error(f"❌ Erreur API : {e}")
        return []

stations = load_stations()
if not stations:
    st.stop()

st.title("💦 Nowcast — Risque d'Inondation")
st.markdown("**Bassin versant de l'Ill - Grand Est**")

# ====================== CARTE ======================
st.subheader("🗺️ Bassin versant de l'Ill (Grand Est)")

df_map = pd.DataFrame(stations)
st.map(
    df_map,
    latitude="latitude",
    longitude="longitude",
    use_container_width=True,
    size=120,
    color="#003189"
)

# ====================== PARAMÈTRES ======================
st.subheader("⚙️ Paramètres")

col1, col2 = st.columns([3, 2])
with col1:
    station_dict = {s["label"]: s["code"] for s in stations}
    selected_name = st.selectbox("Choisir une station", options=list(station_dict.keys()))
    selected_code = station_dict[selected_name]

# Seuil fixe à 2 mètres
ALERT_THRESHOLD_M = 2.0
st.info(f"**Seuil d'alerte fixé à {ALERT_THRESHOLD_M} mètres**")

col_d1, col_d2 = st.columns(2)
with col_d1:
    start_date = st.date_input("Date début", value=datetime(2025, 6, 1))
with col_d2:
    end_date = st.date_input("Date fin", value=datetime.today().date() + timedelta(days=30))

# ====================== GRAPH ======================
st.subheader(f"📈 Hauteur d'eau — {selected_name}")

st.info("🔄 Chargement des observations et prédictions en cours...")

st.warning("Aucune donnée disponible pour le moment. L'API est en cours de connexion.")

st.caption("Données : Hub'eau API — Mastère Architecte IA — Fatimatou Bah — 2026")

st.markdown("---")