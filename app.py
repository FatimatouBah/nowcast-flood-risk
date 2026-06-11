import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="💦 Nowcast Crues", page_icon="💦", layout="wide")

# ====================== CONFIG ======================
API_BASE_URL = st.secrets.get("API_BASE_URL", "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space")

# ====================== FONCTIONS ======================
@st.cache_data(ttl=3600)
def load_stations():
    try:
        r = requests.get(f"{API_BASE_URL}/stations", timeout=10)
        r.raise_for_status()
        return r.json().get("stations", [])
    except Exception as e:
        st.error(f"❌ Impossible de charger les stations : {str(e)}")
        return []

@st.cache_data
def load_station_thresholds(station_code):
    try:
        r = requests.get(f"{API_BASE_URL}/values/hixnj/thresholds", 
                        params={"station_code": station_code}, timeout=10)
        r.raise_for_status()
        thresh = r.json().get("thresholds", {})
        return thresh.get("q0", 0), thresh.get("q98", 2000), thresh.get("q100", 4000)
    except:
        return 0, 2000, 4000

@st.cache_data
def load_station_heights(station_code, from_date, to_date):
    try:
        r = requests.get(f"{API_BASE_URL}/station/hixnj/observations",
                        params={"station_code": station_code, "from_date": from_date, "to_date": to_date}, timeout=15)
        r.raise_for_status()
        return r.json().get("observations", [])
    except:
        return []

@st.cache_data
def predict_station_heights(station_code, from_date, to_date):
    try:
        r = requests.post(f"{API_BASE_URL}/station/hixnj/predict",
                         json={"station_code": station_code, "from_date": from_date, "to_date": to_date}, timeout=15)
        r.raise_for_status()
        return r.json().get("predictions", [])
    except:
        return []

# ====================== INTERFACE ======================
st.title("💦 Suivi des Crues et Prévisions")
st.markdown("**Bassin versant de l'Ill - Grand Est**")

stations = load_stations()
if not stations:
    st.stop()

# ====================== CARTE BASSIN VERSANT ======================
st.subheader("📍 Bassin versant de l'Ill (Grand Est)")

lats, lons, labels, sizes, colors = [], [], [], [], []
for s in stations:
    lat = s.get("latitude")
    lon = s.get("longitude")
    label = s.get("label", "Station")
    code = s.get("code")
    
    lats.append(lat)
    lons.append(lon)
    labels.append(label)
    
    # Mise en évidence de la station La Lauch à Colmar
    if code == "A158020101":
        sizes.append(18)
        colors.append("#003189")
    else:
        sizes.append(11)
        colors.append("#555555")

fig_map = go.Figure(go.Scattermap(
    lat=lats, lon=lons,
    mode="markers+text",
    marker=dict(size=sizes, color=colors, opacity=0.9),
    text=labels,
    textposition="top right",
    hoverinfo="text"
))

fig_map.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=48.25, lon=7.45),
        zoom=8.5
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    height=480
)
st.plotly_chart(fig_map, use_container_width=True)

# ====================== SÉLECTIONS ======================
st.markdown("---")
col1, col2 = st.columns([3, 2])

with col1:
    station_dict = {s["label"]: s["code"] for s in stations}
    selected_name = st.selectbox("**Sélectionner une station**", options=list(station_dict.keys()))
    selected_code = station_dict[selected_name]

with col2:
    alert_threshold_m = st.number_input(
        "Seuil d'alerte (mètres)", 
        value=2.0, 
        min_value=0.5, 
        max_value=10.0, 
        step=0.1
    )

# Dates
today = datetime.today().date()
col_d1, col_d2 = st.columns(2)
with col_d1:
    start_date = st.date_input("Date de début", value=today - timedelta(days=30), min_value=datetime(2007,1,1).date())
with col_d2:
    end_date = st.date_input("Date de fin", value=today + timedelta(days=30), max_value=today + timedelta(days=90))

# ====================== GRAPH ======================
st.markdown("---")
st.subheader("📈 Hauteur d'eau maximale journalière - Observation & Prédiction")

obs = load_station_heights(selected_code, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
pred = predict_station_heights(selected_code, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

fig = go.Figure()
df_obs = pd.DataFrame(obs)
df_pred = pd.DataFrame(pred)

if not df_obs.empty:
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df_obs["ds"]),
        y=df_obs["yobs"],
        name="Observation",
        line=dict(color="#444444", width=2.8)
    ))

if not df_pred.empty:
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df_pred["ds"]),
        y=df_pred["yhat"],
        name="Prédiction",
        line=dict(color="#888888", width=2.5, dash="dash")
    ))

fig.add_hline(
    y=alert_threshold_m * 1000,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Seuil d'alerte ({alert_threshold_m} m)"
)

fig.update_layout(
    title=f"{selected_name}",
    xaxis_title="Date",
    yaxis_title="Hauteur (mm)",
    plot_bgcolor="rgba(240,240,240,0.8)",
    height=520,
    xaxis=dict(rangeslider=dict(visible=True), type="date")
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Données Hub'Eau • Modèle ML • Bassin versant de l'Ill")