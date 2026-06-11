import requests
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space"
st.set_page_config(page_title="Nowcast Inondation", page_icon="💦", layout="wide")

# ─── FONCTION API ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def fetch_data(endpoint, params=None):
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params, timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

# ─── INTERFACE ────────────────────────────────────────────────────────────────

st.title("💦 Nowcast — Risque d'Inondation")
st.markdown("---")

data = fetch_data("stations")
if data and "stations" in data:
    stations = data["stations"]
    station_map = {s["label"]: s for s in stations}
    
    # 1. CARTE (Focus bassin)
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

    # 2. ANALYSE
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
        obs_data = fetch_data("station/hixnj/observations", {"station_code": sel_code, "from_date": start, "to_date": end})
        
        # Vérification robuste : on s'assure que les données existent et ne sont pas vides
        if obs_data and "observations" in obs_data and len(obs_data["observations"]) > 0:
            df = pd.DataFrame(obs_data["observations"])
            
            # S'assure que les colonnes 'ds' et 'yobs' existent, sinon prend les 2 premières
            x_col = 'ds' if 'ds' in df.columns else df.columns[0]
            y_col = 'yobs' if 'yobs' in df.columns else df.columns[1]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.to_datetime(df[x_col]), y=df[y_col], 
                                     mode="lines", name="Observation", line=dict(color="grey", width=2)))
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Seuil Alerte")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=400,
                              xaxis=dict(rangeslider=dict(visible=True), type="date", gridcolor="#e5e5e5"),
                              yaxis=dict(gridcolor="#e5e5e5"))
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Aucune donnée disponible pour cette sélection. Vérifiez la période ou la station.")
else:
    st.error("Impossible de joindre le serveur de données.")

st.caption("Données : Hub'eau API — Mastère Architecte IA — 2026 — Fatimatou Bah")