import requests
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

API_BASE_URL = "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space"

st.set_page_config(page_title="Nowcast Inondation — Ill Grand Est", page_icon="💦", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #003189; }
</style>
""", unsafe_allow_html=True)

# ─── CHARGEMENT API ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_stations():
    r = requests.get(f"{API_BASE_URL}/stations", timeout=10)
    r.raise_for_status()
    return r.json()["stations"]

@st.cache_data(ttl=300)
def load_station_dates(station_code):
    r = requests.get(f"{API_BASE_URL}/values/hixnj/dates", params={"station_code": station_code}, timeout=5)
    r.raise_for_status()
    dates = r.json()["dates"]
    return dates["lower"], dates["upper"]

@st.cache_data(ttl=300)
def load_station_thresholds(station_code):
    r = requests.get(f"{API_BASE_URL}/values/hixnj/thresholds", params={"station_code": station_code}, timeout=5)
    r.raise_for_status()
    t = r.json()["thresholds"]
    return t["q0"], t["q100"], t["q98"]

@st.cache_data(ttl=300)
def load_observations(station_code, from_date, to_date):
    r = requests.get(f"{API_BASE_URL}/station/hixnj/observations",
                     params={"station_code": station_code, "from_date": from_date, "to_date": to_date}, timeout=5)
    r.raise_for_status()
    return r.json()["observations"]

@st.cache_data(ttl=300)
def load_predictions(station_code, from_date, to_date):
    r = requests.post(f"{API_BASE_URL}/station/hixnj/predict",
                      json={"station_code": station_code, "from_date": from_date, "to_date": to_date}, timeout=5)
    r.raise_for_status()
    return r.json()["predictions"]

# ─── INITIALISATION ────────────────────────────────────────────────────────────

st.markdown("# 💦 Nowcast — Risque d'Inondation")
st.markdown("## Bassin versant de l'Ill — Grand Est | Alsace")
st.markdown("---")

# Récupération des stations avant tout usage
try:
    stations = load_stations()
    station_by_code = {s["code"]: s for s in stations}
    station_by_label = {s["label"]: s for s in stations}
except Exception as e:
    st.error(f"Erreur de connexion à l'API : {e}")
    st.stop()

# ─── CARTE ─────────────────────────────────────────────────────────────────────

st.subheader("🗺️ Carte des stations hydrométriques")
today_str = dt.date.today().strftime("%Y-%m-%d")

lats, lons, noms, couleurs, textes = [], [], [], [], []
for s in stations:
    code = s["code"]
    try:
        preds = load_predictions(code, today_str, today_str)
        yhat = preds[-1].get("yhat", 0) if (preds and isinstance(preds, list)) else 0
        t_q0, t_q100, t_q98 = load_station_thresholds(code)
        color = "#28a745" if yhat < t_q98 else "#dc3545"
        niveau = "🟢 Normal" if yhat < t_q98 else "🔴 Alerte"
    except Exception:
        yhat, color, niveau = 0, "#888888", "N/A"

    lats.append(s["latitude"])
    lons.append(s["longitude"])
    noms.append(s["label"])
    couleurs.append(color)
    textes.append(f"<b>{s['label']}</b><br>HIXnJ: {yhat:.0f} mm<br>{niveau}")

fig_map = go.Figure(go.Scattermap(
    lat=lats, lon=lons,
    mode='markers+text',
    marker=dict(size=20, color=couleurs, opacity=0.9),
    text=noms,
    textposition="top right",
    hovertext=textes,
    hoverinfo='text'
))
fig_map.update_layout(
    map=dict(style="carto-positron", center=dict(lat=48.35, lon=7.45), zoom=8.5),
    margin=dict(l=0, r=0, t=0, b=0),
    height=450
)
st.plotly_chart(fig_map, width='stretch')

st.markdown("---")

# ─── RESTE DU CODE (SÉLECTION ET GRAPHIQUES) ───────────────────────────────────

# ... (Tu peux laisser le reste de ton code identique, 
#      assure-toi juste de bien remplacer chaque `use_container_width=True` 
#      par `width='stretch'` dans tes `st.plotly_chart`)

st.caption("Données : Hub'eau API (BRGM) — API ML : nicolaspichon35 — Mastère Architecte IA — Jedha Bootcamp 2026 — Fatimatou Bah")