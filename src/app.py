import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="Nowcast Inondation — Ill Grand Est", page_icon="💦", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin: 5px 0; background: white; border: 1px solid #dee2e6; }
    h1, h2, h3 { color: #003189; }
</style>
""", unsafe_allow_html=True)

# ─── SECRETS ───────────────────────────────────────────────────────────────────
# Sur Hugging Face : Settings → Variables and secrets
# HUBEAU_API_URL = https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr

HUBEAU_API = st.secrets.get("HUBEAU_API_URL", "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr")

# ─── STATIONS ──────────────────────────────────────────────────────────────────
STATIONS = {
    "A161003001": {"nom": "Ill à Colmar",      "lat": 48.080, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 800,  "seuil_max": 2500},
    "A214010001": {"nom": "Fecht à Ostheim",   "lat": 48.166, "lon": 7.358, "seuil_alerte": 2000, "seuil_min": 300,  "seuil_max": 2200},
    "A236003001": {"nom": "Ill à Kogenheim",   "lat": 48.266, "lon": 7.533, "seuil_alerte": 2000, "seuil_min": 400,  "seuil_max": 2000},
    "A348020001": {"nom": "Zorn à Waltenheim", "lat": 48.716, "lon": 7.583, "seuil_alerte": 2000, "seuil_min": 500,  "seuil_max": 1800},
}

# ─── CHARGEMENT MODÈLE ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for root, dirs, files in os.walk("mlruns"):
        for f in files:
            if f == "model.pkl":
                return joblib.load(os.path.join(root, f))
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

# ─── CHARGEMENT DONNÉES LOCALES ────────────────────────────────────────────────
@st.cache_data
def load_local_data():
    dfs = {}
    for path in ["hubeau", "src/fb/hubeau", "/app/hubeau"]:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(".csv") and "obstr" in f and "_H_" in f]
            for f in sorted(files):
                parts = f.replace(".csv","").split("_")
                code = parts[2]
                df = pd.read_csv(os.path.join(path, f))
                df.columns = df.columns.str.lower().str.strip()
                if "date_obs" in df.columns and "resultat_obs" in df.columns:
                    df["resultat_obs"] = pd.to_numeric(df["resultat_obs"], errors="coerce")
                    df["date_obs"] = pd.to_datetime(df["date_obs"])
                    df = df.dropna(subset=["resultat_obs","date_obs"])
                    df = df[(df["resultat_obs"] > 0) & (df["resultat_obs"] < 5000)]
                    dfs[code] = df.sort_values("date_obs")
            if dfs:
                break
    return dfs

# ─── CHARGEMENT DONNÉES HISTORIQUES HUB'EAU ───────────────────────────────────
@st.cache_data(ttl=3600)
def load_hubeau_history(code_station, date_debut, date_fin):
    try:
        url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
        params = {
            "code_entite": code_station,
            "grandeur_hydro_elab": "QmnJ",
            "date_debut_obs_elab": date_debut.strftime("%Y-%m-%d"),
            "date_fin_obs_elab": date_fin.strftime("%Y-%m-%d"),
            "size": 2000,
            "fields": "date_obs_elab,resultat_obs_elab"
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code in [200, 206]:
            data = r.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                df["date_obs_elab"] = pd.to_datetime(df["date_obs_elab"])
                df["resultat_obs_elab"] = pd.to_numeric(df["resultat_obs_elab"], errors="coerce")
                return df.dropna().sort_values("date_obs_elab")
    except Exception:
        pass
    return None

# ─── PRÉDICTION ────────────────────────────────────────────────────────────────
def get_risk(hauteur, heure, mois, model):
    if model is None or hauteur == 0:
        return 1
    X = pd.DataFrame([[hauteur, heure, mois]], columns=["resultat_obs", "hour", "month"])
    pred = int(model.predict(X)[0])
    return max(1, min(pred, 3))

risk_labels = {1: "🟢 Faible", 2: "🟡 Modéré", 3: "🔴 Élevé"}
risk_colors_map = {1: "green", 2: "orange", 3: "red"}

# ─── CHARGEMENT ────────────────────────────────────────────────────────────────
model = load_model()
dfs_local = load_local_data()

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 💦 Nowcast — Risque d'Inondation")
st.markdown("### Bassin versant de l'Ill — Grand Est | Alsace")
st.markdown("---")

# ─── CARTE ─────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Stations hydrométriques — Bassin de l'Ill")

lats, lons, noms, couleurs, textes, tailles = [], [], [], [], [], []
for code, info in STATIONS.items():
    if code in dfs_local and not dfs_local[code].empty:
        h = float(dfs_local[code]["resultat_obs"].iloc[-1])
        hr = dfs_local[code]["date_obs"].iloc[-1].hour
        mo = dfs_local[code]["date_obs"].iloc[-1].month
        risk = get_risk(h, hr, mo, model)
        date_str = dfs_local[code]["date_obs"].iloc[-1].strftime("%d/%m/%Y %H:%M")
    else:
        h, risk, date_str = 0, 1, "N/A"

    lats.append(info["lat"])
    lons.append(info["lon"])
    noms.append(info["nom"])
    couleurs.append(risk_colors_map[risk])
    tailles.append(25)
    textes.append(
        f"<b>{info['nom']}</b><br>"
        f"Hauteur: {h:.0f} mm<br>"
        f"Risque: {risk_labels[risk]}<br>"
        f"Seuil alerte: {info['seuil_alerte']} mm<br>"
        f"Dernière obs: {date_str}"
    )

fig_map = go.Figure(go.Scattermapbox(
    lat=lats, lon=lons,
    mode='markers+text',
    marker=dict(size=tailles, color=couleurs, opacity=0.85),
    text=noms,
    textposition="top right",
    textfont=dict(size=13, color="black"),
    hovertext=textes,
    hoverinfo='text'
))
fig_map.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=48.35, lon=7.45),
        zoom=8.5
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=450,
    paper_bgcolor="white"
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# ─── SÉLECTION STATION ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ Paramètres")
    selected_nom = st.selectbox(
        "Choisir une station",
        [info["nom"] for info in STATIONS.values()]
    )
    selected_code = [c for c, i in STATIONS.items() if i["nom"] == selected_nom][0]
    info_station = STATIONS[selected_code]

    st.markdown("**📅 Période d'analyse**")
    date_min = datetime(2007, 1, 1)
    date_max = datetime.now()
    date_debut = st.date_input("Date début", value=datetime(2020, 1, 1), min_value=date_min, max_value=date_max)
    date_fin = st.date_input("Date fin", value=date_max, min_value=date_min, max_value=date_max)

    st.markdown("**🔮 Date de prévision**")
    date_prevision = st.date_input(
        "Date de prévision",
        value=datetime.now(),
        min_value=datetime.now(),
        max_value=datetime.now() + timedelta(days=90)
    )

    st.markdown("**⚠️ Seuils de la station**")
    seuil_alerte = st.number_input("Seuil d'alerte (mm)", value=info_station["seuil_alerte"], step=50)
    seuil_min_val = st.number_input("Seuil minimum (mm)", value=info_station["seuil_min"], step=50)
    seuil_max_val = st.number_input("Seuil maximum (mm)", value=info_station["seuil_max"], step=50)

with col_right:
    st.subheader(f"📈 Hauteur d'eau maximale journalière — {selected_nom}")

    # Données locales temps réel
    df_local = dfs_local.get(selected_code)

    # Données historiques Hub'eau
    df_hist = load_hubeau_history(selected_code, date_debut, date_fin)

    fig_ts = go.Figure()

    # Données historiques
    if df_hist is not None and not df_hist.empty:
        fig_ts.add_trace(go.Scatter(
            x=df_hist["date_obs_elab"],
            y=df_hist["resultat_obs_elab"],
            mode='lines',
            name="Historique (Hub'eau)",
            line=dict(color='grey', width=1.5),
            opacity=0.8
        ))

    # Données temps réel locales
    if df_local is not None and not df_local.empty:
        df_filtered = df_local[
            (df_local["date_obs"] >= pd.Timestamp(date_debut)) &
            (df_local["date_obs"] <= pd.Timestamp(date_fin))
        ]
        if not df_filtered.empty:
            fig_ts.add_trace(go.Scatter(
                x=df_filtered["date_obs"],
                y=df_filtered["resultat_obs"],
                mode='lines',
                name="Observation temps réel",
                line=dict(color='steelblue', width=2)
            ))

    # Seuils
    fig_ts.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", line_width=2,
                     annotation_text=f"Seuil alerte ({seuil_alerte} mm)",
                     annotation_position="top left")
    fig_ts.add_hline(y=seuil_min_val, line_dash="dot", line_color="green", line_width=1,
                     annotation_text=f"Seuil min ({seuil_min_val} mm)",
                     annotation_position="bottom left")
    fig_ts.add_hline(y=seuil_max_val, line_dash="dot", line_color="darkred", line_width=1,
                     annotation_text=f"Seuil max ({seuil_max_val} mm)",
                     annotation_position="top left")

    fig_ts.update_layout(
        xaxis_title="Date",
        yaxis_title="Hauteur d'eau (mm)",
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(gridcolor="#eeeeee")
    )
    st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# ─── PRÉDICTION ────────────────────────────────────────────────────────────────
st.subheader("🔮 Simulateur de prédiction")

col3, col4, col5 = st.columns(3)
with col3:
    hauteur_sim = st.slider(
        "Hauteur d'eau (mm)",
        int(info_station["seuil_min"]),
        int(info_station["seuil_max"]),
        int((info_station["seuil_min"] + info_station["seuil_max"]) // 2)
    )
with col4:
    heure_sim = st.slider("Heure", 0, 23, datetime.now().hour)
with col5:
    mois_sim = st.slider("Mois", 1, 12, datetime.now().month)

pred_sim = get_risk(hauteur_sim, heure_sim, mois_sim, model)
label_sim = risk_labels[pred_sim]
color_sim = {"🟢 Faible": "#d4edda", "🟡 Modéré": "#fff3cd", "🔴 Élevé": "#f8d7da"}[label_sim]
border_sim = {"🟢 Faible": "#28a745", "🟡 Modéré": "#ffc107", "🔴 Élevé": "#dc3545"}[label_sim]
text_sim = {"🟢 Faible": "#155724", "🟡 Modéré": "#856404", "🔴 Élevé": "#721c24"}[label_sim]

col_pred, col_info = st.columns([1, 2])
with col_pred:
    st.markdown(f"""
    <div style='background:{color_sim}; border:2px solid {border_sim}; color:{text_sim};
                padding:25px; border-radius:12px; text-align:center; font-size:1.8em; font-weight:bold'>
        {label_sim}
    </div>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown(f"""
    **Station :** {selected_nom}  
    **Hauteur simulée :** {hauteur_sim} mm  
    **Seuil d'alerte :** {seuil_alerte} mm  
    **Dépassement seuil :** {"⚠️ OUI" if hauteur_sim >= seuil_alerte else "✅ NON"}  
    **Date de prévision :** {date_prevision.strftime('%d/%m/%Y')}  
    **Modèle :** Random Forest (F1 = 0.98)  
    """)

st.markdown("---")
st.caption("Données : Hub'eau API (BRGM) — Modèle ML : Random Forest — Mastère Architecte IA — Jedha Bootcamp 2026 — Fatimatou Bah")