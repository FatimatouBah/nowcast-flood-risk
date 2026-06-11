import requests
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

API_BASE_URL = "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space"

st.set_page_config(page_title="Nowcast Inondation", page_icon="💦", layout="wide")

# ─── FONCTIONS API ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_stations():
    r = requests.get(f"{API_BASE_URL}/stations")
    r.raise_for_status()
    return r.json()["stations"]

@st.cache_data(ttl=300)
def load_observations(station_code, from_date, to_date):
    r = requests.get(f"{API_BASE_URL}/station/hixnj/observations",
                     params={"station_code": station_code, "from_date": from_date, "to_date": to_date})
    return pd.DataFrame(r.json()["observations"])

# ─── INTERFACE ─────────────────────────────────────────────────────────────────

st.markdown("# 💦 Nowcast — Risque d'Inondation")
st.markdown("---")

stations = load_stations()
station_by_label = {s["label"]: s for s in stations}

# ─── CARTE DES STATIONS ───────────────────────────────────────────────────────

st.subheader("🗺️ Carte des stations")

lats = [s["latitude"] for s in stations]
lons = [s["longitude"] for s in stations]
noms = [s["label"] for s in stations]

fig_map = go.Figure(go.Scattermap(
    lat=lats, lon=lons, mode='markers+text',
    marker=dict(size=18, color="#003189", opacity=0.9),
    text=noms, textposition="top right"
))

fig_map.update_layout(
    map=dict(style="carto-positron", center=dict(lat=48.35, lon=7.45), zoom=8),
    margin=dict(l=0, r=0, t=0, b=0), height=400
)
st.plotly_chart(fig_map, width='stretch')

# ─── ANALYSE STATION ──────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    selected_label = st.selectbox("Station", list(station_by_label.keys()))
    sel_code = station_by_label[selected_label]["code"]
    
    date_debut = st.date_input("Date début", value=dt.date(2024, 1, 1), min_value=dt.date(2007, 1, 1), max_value=dt.date(2026, 12, 31))
    date_fin = st.date_input("Date fin", value=dt.date.today(), min_value=dt.date(2007, 1, 1), max_value=dt.date(2026, 12, 31))
    
    # Surcharge manuelle du seuil
    seuil_alerte = st.number_input("Seuil alerte (m)", value=2.0, step=0.1)

with col2:
    st.subheader(f"📈 Hauteur d'eau — {selected_label}")
    df = load_observations(sel_code, date_debut.strftime("%Y-%m-%d"), date_fin.strftime("%Y-%m-%d"))

    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(df["ds"]), y=df["yobs"], 
                                 mode="lines", name="Observations", line=dict(color="grey", width=2)))
        
        fig.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", annotation_text="Seuil Alerte")
        
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(rangeslider=dict(visible=True), type="date", gridcolor="#e5e5e5"),
            yaxis=dict(gridcolor="#e5e5e5"), height=400
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Aucune donnée disponible pour cette sélection.")

st.caption("Données : Hub'eau API — Mastère Architecte IA — Jedha Bootcamp 2026 — Fatimatou Bah")