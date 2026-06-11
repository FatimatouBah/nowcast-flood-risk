import requests
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

# Configuration du dashboard
API_BASE_URL = "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space"
st.set_page_config(page_title="Nowcast Inondation - Grand Est", page_icon="💦", layout="wide")

# ─── FONCTIONS DE RÉCUPÉRATION ──────────────────────────────────────────────

@st.cache_data(ttl=600)
def fetch_data(endpoint, params=None):
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# ─── INTERFACE ──────────────────────────────────────────────────────────────

st.title("💦 Nowcast — Risque d'Inondation (Grand Est)")
st.markdown("---")

data = fetch_data("stations")
if data and "stations" in data:
    stations = data["stations"]
    station_map = {s["label"]: s for s in stations}

    # Section Carte (Zoom sur le bassin de l'Ill)
    st.subheader("🗺️ Carte des stations")
    lats = [s["latitude"] for s in stations]
    lons = [s["longitude"] for s in stations]
    noms = [s["label"] for s in stations]

    fig_map = go.Figure(go.Scattermap(
        lat=lats, lon=lons, mode='markers+text',
        marker=dict(size=14, color="#003189"),
        text=noms, textposition="top right"
    ))
    fig_map.update_layout(
        map=dict(style="carto-positron", center=dict(lat=48.35, lon=7.45), zoom=9),
        margin=dict(l=0, r=0, t=0, b=0), height=350
    )
    st.plotly_chart(fig_map, width='stretch')

    # Section Analyse
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("⚙️ Paramètres")
        sel_label = st.selectbox("Choisir une station", list(station_map.keys()))
        sel_code = station_map[sel_label]["code"]
        
        start = st.date_input("Date début", dt.date(2015, 1, 1))
        end = st.date_input("Date fin", dt.date.today())
        threshold = st.number_input("Seuil alerte (m)", value=2.0, step=0.1)

    with col2:
        st.subheader(f"📈 Hauteur d'eau — {sel_label}")
        obs = fetch_data("station/hixnj/observations", {"station_code": sel_code, "from_date": start, "to_date": end})
        
        if obs and "observations" in obs:
            df = pd.DataFrame(obs["observations"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.to_datetime(df["ds"]), y=df["yobs"], 
                                     mode="lines", name="Observation", line=dict(color="grey", width=2)))
            
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Seuil Alerte")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=400,
                              xaxis=dict(rangeslider=dict(visible=True), type="date", gridcolor="#e5e5e5"),
                              yaxis=dict(gridcolor="#e5e5e5"))
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Aucune donnée disponible pour cette période.")
else:
    st.error("Connexion à l'API impossible. Vérifiez l'URL ou l'état du serveur.")

st.caption("Données : Hub'eau API — Mastère Architecte IA — Jedha Bootcamp 2026 — Fatimatou Bah")