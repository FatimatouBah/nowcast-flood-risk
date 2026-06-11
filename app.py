import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="💦 Nowcast Crues", page_icon="💦", layout="wide")

API_BASE_URL = st.secrets.get("API_BASE_URL", "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space")

# ====================== FONCTIONS ======================
@st.cache_data(ttl=3600)
def load_stations():
    try:
        r = requests.get(f"{API_BASE_URL}/stations", timeout=10)
        r.raise_for_status()
        return r.json().get("stations", [])
    except Exception as e:
        st.error(f"❌ Erreur stations : {e}")
        return []

@st.cache_data
def load_station_heights(station_code, from_date, to_date):
    try:
        url = f"{API_BASE_URL}/station/hixnj/observations"
        params = {"station_code": station_code, "from_date": from_date, "to_date": to_date}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("observations", [])
    except Exception as e:
        st.warning(f"⚠️ Observations indisponibles : {str(e)[:100]}")
        return []

@st.cache_data
def predict_station_heights(station_code, from_date, to_date):
    try:
        url = f"{API_BASE_URL}/station/hixnj/predict"
        payload = {"station_code": station_code, "from_date": from_date, "to_date": to_date}
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("predictions", [])
    except Exception as e:
        st.warning(f"⚠️ Prédictions indisponibles : {str(e)[:100]}")
        return []

# ====================== APP ======================
st.title("💦 Nowcast — Risque d'Inondation")
st.markdown("**Bassin versant de l'Ill - Grand Est**")

stations = load_stations()
if not stations:
    st.stop()

# Carte
st.subheader("🗺️ Carte des stations")
lats = [s["latitude"] for s in stations]
lons = [s["longitude"] for s in stations]
labels = [s["label"] for s in stations]

fig_map = go.Figure(go.Scattermap(
    lat=lats, lon=lons,
    mode="markers+text",
    marker=dict(size=12, color="#003189"),
    text=labels,
    textposition="top right"
))
fig_map.update_layout(
    mapbox=dict(style="open-street-map", center=dict(lat=48.25, lon=7.45), zoom=8.5),
    height=450
)
st.plotly_chart(fig_map, use_container_width=True)

# Paramètres
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
    start_date = st.date_input("Date début", value=datetime.today().date() - timedelta(days=30))
with col_d2:
    end_date = st.date_input("Date fin", value=datetime.today().date() + timedelta(days=30))

# Graphique
st.subheader(f"📈 Hauteur d'eau — {selected_name}")

obs = load_station_heights(selected_code, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
pred = predict_station_heights(selected_code, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

fig = go.Figure()

if obs:
    df_obs = pd.DataFrame(obs)
    fig.add_trace(go.Scatter(x=pd.to_datetime(df_obs["ds"]), y=df_obs["yobs"], name="Observation", line=dict(color="#444")))
else:
    st.info("Aucune observation disponible pour cette période.")

if pred:
    df_pred = pd.DataFrame(pred)
    fig.add_trace(go.Scatter(x=pd.to_datetime(df_pred["ds"]), y=df_pred["yhat"], name="Prédiction", line=dict(dash="dash", color="#888")))

if obs or pred:
    fig.add_hline(y=alert_threshold_m*1000, line_dash="dash", line_color="red", annotation_text=f"Seuil {alert_threshold_m}m")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Hauteur (mm)",
        height=520,
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("🚫 Aucune donnée disponible pour cette sélection. Vérifiez la période ou essayez une autre station.")

st.caption("Données : Hub'eau API — Mastère Architecte IA — 2026 — Fatimatou Bah")