import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="💦 Nowcast Crues", page_icon="💦", layout="wide")

API_BASE_URL = st.secrets.get("API_BASE_URL", "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space")

@st.cache_data(ttl=3600)
def load_stations():
    try:
        r = requests.get(f"{API_BASE_URL}/stations", timeout=10)
        r.raise_for_status()
        return r.json().get("stations", [])
    except Exception as e:
        st.error(f"❌ Erreur chargement stations : {e}")
        return []

stations = load_stations()
if not stations:
    st.stop()

st.title("💦 Nowcast — Risque d'Inondation")
st.markdown("**Bassin versant de l'Ill - Grand Est**")

# ====================== CARTE OPTIMISÉE ======================
st.subheader("🗺️ Bassin versant de l'Ill (Grand Est)")

try:
    df_stations = pd.DataFrame(stations)
    
    fig = go.Figure()

    # Toutes les stations
    fig.add_trace(go.Scattermap(
        lat=df_stations["latitude"],
        lon=df_stations["longitude"],
        mode="markers+text",
        marker=dict(size=12, color="#555555"),
        text=df_stations["label"],
        textposition="top right",
        hoverinfo="text"
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=48.28, lon=7.42),   # Centre optimisé bassin Ill
            zoom=8.7                            # Zoom bien sur le bassin versant
        ),
        height=520,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning("Problème avec la carte avancée")
    # Fallback simple
    st.map(df_stations, latitude="latitude", longitude="longitude", use_container_width=True)

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
    start_date = st.date_input("Date début", value=datetime(2025, 1, 1))
with col_d2:
    end_date = st.date_input("Date fin", value=datetime.today().date() + timedelta(days=30))

st.caption("Données : Hub'eau API — Grand Est")