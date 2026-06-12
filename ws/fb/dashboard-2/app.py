import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

st.set_page_config(page_title="💦 Nowcast Crues", page_icon="💦", layout="wide")

# Style CSS pour le titre et le fond
st.markdown("""
<style>
    h1, h2, h3 { color: #003189; }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = st.secrets.get("API_BASE_URL", "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space")

# ─── CHARGEMENT API ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_stations():
    try:
        r = requests.get(f"{API_BASE_URL}/stations", timeout=10)
        r.raise_for_status()
        return r.json().get("stations", [])
    except Exception as e:
        st.error(f"❌ Erreur API stations : {e}")
        return []

@st.cache_data(ttl=3600)
def load_station_dates(station_code):
    try:
        r = requests.get(f"{API_BASE_URL}/values/hixnj/dates", params={"station_code": station_code}, timeout=10)
        r.raise_for_status()
        dates = r.json()["dates"]
        return dates["lower"], dates["upper"]
    except Exception:
        return "2007-01-01", date.today().isoformat()

@st.cache_data(ttl=600)
def load_observations(station_code, from_date, to_date):
    try:
        r = requests.get(f"{API_BASE_URL}/station/hixnj/observations",
                         params={"station_code": station_code, "from_date": from_date, "to_date": to_date}, timeout=15)
        r.raise_for_status()
        return r.json().get("observations", [])
    except Exception:
        return []

# ─── INTERFACE ────────────────────────────────────────────────────────────────

st.title("💦 Nowcast — Risque d'Inondation")
st.markdown("**Bassin versant de l'Ill — Grand Est**")
st.markdown("---")

stations = load_stations()
if not stations:
    st.stop()

station_dict = {s["label"]: s for s in stations}

# ─── CARTE ────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Stations hydrométriques")
df_map = pd.DataFrame(stations)
st.map(df_map, latitude="latitude", longitude="longitude", size=150, color="#003189")

st.markdown("---")

# ─── PARAMÈTRES + GRAPHIQUE ──────────────────────────────────────────────────

col_params, col_graph = st.columns([1, 3])

with col_params:
    st.subheader("⚙️ Paramètres")
    selected_name = st.selectbox("Choisir une station", list(station_dict.keys()))
    selected_code = station_dict[selected_name]["code"]

    date_lower, date_upper = load_station_dates(selected_code)
    date_min = datetime.fromisoformat(date_lower[:10]).date()
    date_max = datetime.fromisoformat(date_upper[:10]).date()

    start_obs = st.date_input("Début observations", value=date(2024, 1, 1), min_value=date_min, max_value=date_max)
    end_obs = st.date_input("Fin observations", value=date.today(), min_value=date_min, max_value=date_max)
    
    st.metric("Seuil d'alerte imposé", "2.0 m")

with col_graph:
    st.subheader(f"📈 Hauteur d'eau — {selected_name}")
    obs = load_observations(selected_code, start_obs.isoformat(), end_obs.isoformat())
    df_obs = pd.DataFrame(obs)

    fig = go.Figure()

    if not df_obs.empty and "ds" in df_obs.columns and "yobs" in df_obs.columns:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df_obs["ds"]), y=df_obs["yobs"],
            mode="lines", name="Observations", line=dict(color="#bdc3c7", width=2)
        ))

        # Seuil imposé à 2000 mm (2 mètres)
        fig.add_hline(y=2000, line_dash="dash", line_color="red", line_width=3,
                      annotation_text="Seuil Alerte (2m)", annotation_position="top left",
                      annotation_font_color="white")

    # Mise en forme graphique avec fond sombre pour lisibilité en présentation
    fig.update_layout(
        xaxis_title="Date", yaxis_title="HIXnJ (mm)", height=500,
        plot_bgcolor="#262730", paper_bgcolor="#262730",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#444444", rangeslider=dict(visible=True, bgcolor="#262730")),
        yaxis=dict(gridcolor="#444444")
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Données : Hub'eau API — Mastère Architecte IA — 2026 — Fatimatou Bah")