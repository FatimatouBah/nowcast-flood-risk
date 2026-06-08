import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Nowcast Risque Inondation — Ill Grand Est",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .risk-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .risk-green  { background-color: #d4edda; color: #155724; border: 2px solid #28a745; }
    .risk-yellow { background-color: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .risk-red    { background-color: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    h1 { color: #003189; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌊 Nowcast Risque d'Inondation")
st.markdown("### Bassin versant de l'Ill — Grand Est | Mise à jour en quasi-temps réel")
st.markdown("---")

@st.cache_resource
def load_model():
    for root, dirs, files in os.walk("mlruns"):
        for f in files:
            if f == "model.pkl":
                return joblib.load(os.path.join(root, f))
    raise FileNotFoundError("Aucun modèle trouvé dans mlruns")

@st.cache_data
def load_data():
    possible_paths = [
        "hubeau",
        "src/fb/hubeau",
        "/app/hubeau",
        "/app/src/fb/hubeau"
    ]

    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError("Dossier hubeau introuvable")

    files = [f for f in os.listdir(data_path) if f.endswith(".csv") and "obstr" in f]

    dfs = {}
    for f in sorted(files):
        parts = f.replace(".csv", "").split("_")
        station = parts[2]
        grandeur = parts[3]
        key = f"{station}_{grandeur}"

        df = pd.read_csv(os.path.join(data_path, f))
        df["date_obs"] = pd.to_datetime(df["date_obs"])
        df = df.sort_values("date_obs")

        dfs[key] = df

    return dfs

model = load_model()
dfs = load_data()

stations_info = {
    "A161003001": {"nom": "Ill à Colmar",       "lat": 48.080, "lon": 7.358, "riviere": "Ill"},
    "A214010001": {"nom": "Fecht à Ostheim",     "lat": 48.166, "lon": 7.358, "riviere": "Fecht"},
    "A236003001": {"nom": "Ill à Kogenheim",     "lat": 48.266, "lon": 7.533, "riviere": "Ill"},
    "A348020001": {"nom": "Zorn à Waltenheim",   "lat": 48.716, "lon": 7.583, "riviere": "Zorn"},
}

def get_risk(station_code):
    key = f"{station_code}_H"
    if key not in dfs:
        return 0, 0
    df = dfs[key]
    last = df.iloc[-1]
    hauteur = last['resultat_obs']
    heure = last['date_obs'].hour
    mois = last['date_obs'].month
    X = pd.DataFrame([[hauteur, heure, mois]], columns=["resultat_obs", "hour", "month"])
    pred = model.predict(X)[0]
    return int(pred), hauteur

risk_colors = {1: "#28a745", 2: "#ffc107", 3: "#dc3545"}
risk_labels = {1: "🟢 FAIBLE", 2: "🟡 MODÉRÉ", 3: "🔴 ÉLEVÉ"}
risk_css    = {1: "risk-green", 2: "risk-yellow", 3: "risk-red"}

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🗺️ Carte des stations — Bassin de l'Ill")
    lats, lons, noms, couleurs, textes = [], [], [], [], []
    for code, info in stations_info.items():
        risk, hauteur = get_risk(code)
        lats.append(info["lat"])
        lons.append(info["lon"])
        noms.append(info["nom"])
        couleurs.append(risk_colors.get(risk, "#888888"))
        textes.append(f"{info['nom']}<br>Risque: {risk_labels.get(risk,'N/A')}<br>Hauteur: {hauteur:.0f} mm")

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
        mapbox=dict(style="open-street-map", center=dict(lat=48.3, lon=7.45), zoom=8),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420
    )
    st.plotly_chart(fig_map, width='stretch')

with col2:
    st.subheader("🚨 Niveaux de vigilance")
    for code, info in stations_info.items():
        risk, hauteur = get_risk(code)
        css = risk_css.get(risk, "risk-green")
        label = risk_labels.get(risk, "N/A")
        st.markdown(f"""
        <div class='risk-card {css}'>
            {info['nom']}<br>
            <span style='font-size:1.2em'>{label}</span><br>
            <small style='font-weight:normal'>Hauteur actuelle: {hauteur:.0f} mm</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("📈 Séries temporelles — Hauteur d'eau par station")

station_options = {info["nom"]: code for code, info in stations_info.items()}
selected_nom = st.selectbox("Choisir une station", list(station_options.keys()))
selected_code = station_options[selected_nom]

key_h = f"{selected_code}_H"
if key_h in dfs:
    df_plot = dfs[key_h].copy()
    q33 = df_plot['resultat_obs'].quantile(0.33)
    q66 = df_plot['resultat_obs'].quantile(0.66)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_plot['date_obs'], y=df_plot['resultat_obs'],
        mode='lines', name="Hauteur d'eau",
        line=dict(color='#003189', width=2)
    ))
    fig_ts.add_hline(y=q33, line_dash="dash", line_color="#28a745", annotation_text="Seuil faible→modéré")
    fig_ts.add_hline(y=q66, line_dash="dash", line_color="#dc3545", annotation_text="Seuil modéré→élevé")
    fig_ts.update_layout(
        title=f"Hauteur d'eau — {stations_info[selected_code]['nom']}",
        xaxis_title="Date",
        yaxis_title="Hauteur (mm)",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    st.plotly_chart(fig_ts, width='stretch')

st.markdown("---")
st.subheader("🔮 Simulateur de prédiction")
col3, col4, col5 = st.columns(3)
with col3:
    hauteur_sim = st.slider("Hauteur d'eau (mm)", 400, 1500, 800)
with col4:
    heure_sim = st.slider("Heure", 0, 23, 12)
with col5:
    mois_sim = st.slider("Mois", 1, 12, 6)

X_sim = pd.DataFrame([[hauteur_sim, heure_sim, mois_sim]], columns=["resultat_obs", "hour", "month"])
pred_sim = int(model.predict(X_sim)[0])
css_sim = risk_css.get(pred_sim, "risk-green")
label_sim = risk_labels.get(pred_sim, "N/A")

st.markdown(f"""
<div class='risk-card {css_sim}' style='font-size:2em; padding:30px'>
    Niveau de risque prédit : {label_sim}
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Données : Hub'eau API — Modèle : Random Forest — Mastère Architecte IA — Jedha Bootcamp 2026")