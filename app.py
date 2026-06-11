import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

st.set_page_config(page_title="💦 Nowcast Crues", page_icon="💦", layout="wide")

API_BASE_URL = st.secrets.get("API_BASE_URL", "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space")

st.markdown("""
<style>
    h1, h2, h3 { color: #003189; }
    .main { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

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
        r = requests.get(f"{API_BASE_URL}/values/hixnj/dates",
                         params={"station_code": station_code}, timeout=10)
        r.raise_for_status()
        dates = r.json()["dates"]
        return dates["lower"], dates["upper"]
    except Exception:
        return "2007-01-01", date.today().isoformat()

@st.cache_data(ttl=3600)
def load_station_thresholds(station_code):
    try:
        r = requests.get(f"{API_BASE_URL}/values/hixnj/thresholds",
                         params={"station_code": station_code}, timeout=10)
        r.raise_for_status()
        t = r.json()["thresholds"]
        return t["q0"], t["q100"], t["q98"]
    except Exception:
        return 0, 5000, 2000

@st.cache_data(ttl=600)
def load_observations(station_code, from_date, to_date):
    try:
        r = requests.get(f"{API_BASE_URL}/station/hixnj/observations",
                         params={"station_code": station_code,
                                 "from_date": from_date,
                                 "to_date": to_date}, timeout=15)
        r.raise_for_status()
        return r.json().get("observations", [])
    except Exception as e:
        st.warning(f"⚠️ Observations indisponibles : {e}")
        return []

@st.cache_data(ttl=600)
def load_predictions(station_code, from_date, to_date):
    try:
        r = requests.post(f"{API_BASE_URL}/station/hixnj/predict",
                          json={"station_code": station_code,
                                "from_date": from_date,
                                "to_date": to_date}, timeout=30)
        r.raise_for_status()
        return r.json().get("predictions", [])
    except Exception as e:
        st.warning(f"⚠️ Prédictions indisponibles : {e}")
        return []

# ─── INTERFACE ─────────────────────────────────────────────────────────────────

st.title("💦 Nowcast — Risque d'Inondation")
st.markdown("**Bassin versant de l'Ill — Grand Est | Alsace**")
st.markdown("---")

stations = load_stations()
if not stations:
    st.stop()

station_dict = {s["label"]: s for s in stations}

# ─── CARTE ─────────────────────────────────────────────────────────────────────

st.subheader("🗺️ Stations hydrométriques — Bassin de l'Ill")

today_str = date.today().isoformat()

lats, lons, noms, couleurs, textes = [], [], [], [], []
for s in stations:
    code = s["code"]
    try:
        preds = load_predictions(code, today_str, today_str)
        yhat = preds[-1]["yhat"] if preds else 0
        _, _, q98 = load_station_thresholds(code)
        color = "#dc3545" if yhat >= q98 else "#28a745"
        niveau = "🔴 Alerte" if yhat >= q98 else "🟢 Normal"
    except Exception:
        yhat, color, niveau = 0, "#888888", "N/A"

    lats.append(s["latitude"])
    lons.append(s["longitude"])
    noms.append(s["label"])
    couleurs.append(color)
    textes.append(f"<b>{s['label']}</b><br>HIXnJ: {yhat:.0f} mm<br>{niveau}")

fig_map = go.Figure(go.Scattermapbox(
    lat=lats, lon=lons,
    mode='markers+text',
    marker=dict(size=22, color=couleurs, opacity=0.9),
    text=noms,
    textposition="top right",
    textfont=dict(size=13, color="black"),
    hovertext=textes,
    hoverinfo='text'
))
fig_map.update_layout(
    mapbox=dict(style="carto-positron", center=dict(lat=48.35, lon=7.45), zoom=8.5),
    margin=dict(l=0, r=0, t=0, b=0),
    height=430,
    paper_bgcolor="white"
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# ─── PARAMÈTRES + GRAPHIQUE ────────────────────────────────────────────────────

col_params, col_graph = st.columns([1, 3])

with col_params:
    st.subheader("⚙️ Paramètres")

    selected_name = st.selectbox("Choisir une station", list(station_dict.keys()))
    selected_code = station_dict[selected_name]["code"]

    date_lower, date_upper = load_station_dates(selected_code)
    date_min = datetime.fromisoformat(date_lower[:10]).date()
    date_max = datetime.fromisoformat(date_upper[:10]).date()

    st.markdown(f"📅 Données disponibles : **{date_min}** → **{date_max}**")

    start_obs = st.date_input("Début observations", value=date(2024, 1, 1),
                              min_value=date_min, max_value=date_max)
    end_obs = st.date_input("Fin observations", value=date.today(),
                            min_value=date_min, max_value=date_max)

    st.markdown("---")
    st.markdown("**🔮 Prévisions**")
    end_pred = st.date_input("Fin prévision",
                             value=date.today() + timedelta(days=30),
                             min_value=date.today(),
                             max_value=date.today() + timedelta(days=90))

    st.markdown("---")
    q0, q100, q98 = load_station_thresholds(selected_code)
    st.markdown("**⚠️ Seuils de la station**")
    st.markdown(f"""
    - 🟢 Min (Q0) : **{q0:.0f} mm**
    - 🟡 Alerte (Q98) : **{q98:.0f} mm**
    - 🔴 Max (Q100) : **{q100:.0f} mm**
    """)

with col_graph:
    st.subheader(f"📈 Hauteur d'eau maximale journalière — {selected_name}")

    obs = load_observations(selected_code, start_obs.isoformat(), end_obs.isoformat())
    preds = load_predictions(selected_code, end_obs.isoformat(), end_pred.isoformat())

    df_obs = pd.DataFrame(obs)
    df_pred = pd.DataFrame(preds)

    fig = go.Figure()

    if not df_obs.empty and "ds" in df_obs.columns and "yobs" in df_obs.columns:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df_obs["ds"]),
            y=df_obs["yobs"],
            mode="lines",
            name="Observations",
            line=dict(color="grey", width=2)
        ))

    if not df_pred.empty and "ds" in df_pred.columns and "yhat" in df_pred.columns:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df_pred["ds"]),
            y=df_pred["yhat"],
            mode="lines",
            name="Prévisions",
            line=dict(color="#003189", width=2, dash="dash")
        ))

    fig.add_hline(y=q98, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"Seuil alerte Q98 ({q98:.0f} mm)",
                  annotation_position="top left")
    fig.add_hline(y=q0, line_dash="dot", line_color="green", line_width=1,
                  annotation_text=f"Min Q0 ({q0:.0f} mm)",
                  annotation_position="bottom left")
    fig.add_hline(y=q100, line_dash="dot", line_color="darkred", line_width=1,
                  annotation_text=f"Max Q100 ({q100:.0f} mm)",
                  annotation_position="top left")

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="HIXnJ (mm)",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        yaxis=dict(gridcolor="#eeeeee")
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Données : Hub'eau API (BRGM) — API ML : nicolaspichon35 — Mastère Architecte IA — Jedha Bootcamp 2026 — Fatimatou Bah")