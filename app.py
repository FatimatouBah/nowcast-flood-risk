import requests
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space"
st.set_page_config(page_title="Nowcast Inondation", page_icon="💦", layout="wide")

# ─── FONCTIONS API ROBUSTES ────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def fetch_data(endpoint, params=None):
    try:
        r = requests.get(f"{API_BASE_URL}/{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API ({endpoint}): {e}")
        return None

# ─── INTERFACE ─────────────────────────────────────────────────────────────────

st.markdown("# 💦 Nowcast — Risque d'Inondation")
st.markdown("---")

# Récupération stations
data_stations = fetch_data("stations")
if data_stations:
    stations = data_stations.get("stations", [])
    station_map = {s["label"]: s for s in stations}
    
    # 1. CARTE (Optimisée)
    st.subheader("🗺️ Carte des stations")
    lats = [s["latitude"] for s in stations]
    lons = [s["longitude"] for s in stations]
    
    fig_map = go.Figure(go.Scattermap(
        lat=lats, lon=lons, mode='markers',
        marker=dict(size=15, color="#003189")
    ))
    fig_map.update_layout(map=dict(style="carto-positron", center=dict(lat=48.35, lon=7.45), zoom=8), 
                          margin=dict(l=0,r=0,t=0,b=0), height=300)
    st.plotly_chart(fig_map, width='stretch')

    # 2. ANALYSE (Paramètres)
    col1, col2 = st.columns([1, 3])
    with col1:
        sel_label = st.selectbox("Station", list(station_map.keys()))
        code = station_map[sel_label]["code"]
        start = st.date_input("Date début", dt.date(2015, 1, 1))
        end = st.date_input("Date fin", dt.date.today())
        threshold = st.number_input("Seuil alerte (m)", value=2.0)

    with col2:
        st.subheader(f"📈 Analyse : {sel_label}")
        obs_json = fetch_data("station/hixnj/observations", {"station_code": code, "from_date": start, "to_date": end})
        
        if obs_json and "observations" in obs_json:
            df = pd.DataFrame(obs_json["observations"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.to_datetime(df["ds"]), y=df["yobs"], line=dict(color="grey"), name="Observation"))
            fig.add_hline(y=threshold, line_dash="dash", line_color="red")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=400,
                              xaxis=dict(rangeslider=dict(visible=True), type="date"))
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Pas de données trouvées pour cette période. Essayez d'élargir les dates.")

else:
    st.error("Impossible de joindre le serveur de données.")

st.caption("Données : Hub'eau API — Mastère Architecte IA — 2026")