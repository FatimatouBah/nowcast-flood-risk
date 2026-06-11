import requests
from typing import Optional
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------

# # called to load the model into the global object "model" : 
# # ```model = load_model()``` 
# # return a model object (with 'predict' method).
# @st.cache_resource
# def load_model():
#     for root, dirs, files in os.walk("mlruns"):
#         for f in files:
#             if f == "model.pkl":
#                 return joblib.load(os.path.join(root, f))
#     raise FileNotFoundError("Aucun modèle trouvé dans mlruns")

# # called to load the data into the global object "dfs" : 
# # ```dfs = load_data()```.
# # return ???.
# @st.cache_data
# def load_data():
#     possible_paths = [
#         "hubeau",
#         "src/fb/hubeau",
#         "/app/hubeau",
#         "/app/src/fb/hubeau"
#     ]
#     data_path = None
#     for path in possible_paths:
#         if os.path.exists(path):
#             data_path = path
#             break
#     if data_path is None:
#         raise FileNotFoundError("Dossier hubeau introuvable")
#     files = [f for f in os.listdir(data_path) if f.endswith(".csv") and "obstr" in f]
#     dfs = {}
#     for f in sorted(files):
#         parts = f.replace(".csv", "").split("_")
#         station = parts[2]
#         grandeur = parts[3]
#         key = f"{station}_{grandeur}"
#         df = pd.read_csv(os.path.join(data_path, f))
#         df["date_obs"] = pd.to_datetime(df["date_obs"])
#         df = df.sort_values("date_obs")
#         dfs[key] = df
#     return dfs

# URL de l'API (à adapter une fois déployée sur Hugging Face)
API_BASE_URL = "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space"

# Dans app.py, AVANT load_stations(), hors de tout cache
def _wake_up_api():
    try:
        requests.get(f"{API_BASE_URL}/status", timeout=60)
    except Exception:
        pass  # Silencieux, juste pour réveiller le Space

_wake_up_api()  # ligne ~190, juste avant : stations = load_stations()

# ...
@st.cache_data(ttl=3600)
def load_stations():
       
    # TMP :
    # stations = {
    #     "A161003001": {"nom": "Ill à Colmar",        "lat": 48.080, "lon": 7.358, "riviere": "Ill"},
    #     "A214010001": {"nom": "Fecht à Ostheim",     "lat": 48.166, "lon": 7.358, "riviere": "Fecht"},
    #     "A236003001": {"nom": "Ill à Kogenheim",     "lat": 48.266, "lon": 7.533, "riviere": "Ill"},
    #     "A348020001": {"nom": "Zorn à Waltenheim",   "lat": 48.716, "lon": 7.583, "riviere": "Zorn"},
    # }

    url = f"{API_BASE_URL}/stations"
    response = requests.get(url)
    response.raise_for_status()
    content = response.json()
    # DEBUG : print("****** content:", content)
    assert(isinstance(content, dict))
    assert(all(field in content.keys() for field in ["api_version", "count", "stations"]))
    stations = content["stations"]
    # DEBUG : print("****** stations:", stations)
    assert(isinstance(stations, list))
    assert((len(stations) == 0) or (isinstance(stations[0], dict) and (all(field in stations[0].keys() for field in ["code", "label", "latitude", "longitude"]))))
    return stations

# ...
@st.cache_data(ttl=3600)
def load_station_dates_and_stats(station_code):
    url = f"{API_BASE_URL}/station/hixnj/stats"
    response = requests.get(url, params={"station_code": station_code})
    response.raise_for_status()
    content = response.json()
    # DEBUG : print("****** content:", content)
    assert(isinstance(content, dict))
    assert(all(field in content.keys() for field in ["dates", "stats"]))
    dates = content["dates"]
    # DEBUG : print("****** dates:", dates)
    assert(isinstance(dates, dict))
    assert(all(field in dates.keys() for field in ["lower", "upper"]))
    stats = content["stats"]
    # DEBUG : print("****** stats:", stats)
    assert(isinstance(stats, dict))
    assert(all(field in stats.keys() for field in ["min", "max", "q98"]))
    return dates, stats

# ...
@st.cache_data(ttl=3600)
def load_station_heights(station_code, from_date: Optional[str] = None, to_date: Optional[str] = None):
    url = f"{API_BASE_URL}/station/hixnj/observations"
    params = params={"station_code": station_code}
    if from_date is not None:
        params["from_date"] = from_date
    if to_date is not None:
        params["to_date"] = to_date
    response = requests.get(url, params=params)
    response.raise_for_status()
    content = response.json()
    # DEBUG : print("****** content:", content)
    assert(isinstance(content, dict))
    assert(all(field in content.keys() for field in ["api_version", "count", "observations"]))
    observations = content["observations"]
    # DEBUG : print("****** height observations:", observations)
    assert(isinstance(observations, list))
    assert((len(observations) == 0) or (isinstance(observations[0], dict) and (all(field in observations[0].keys() for field in ["ds", "yobs"]))))
    return observations

# ...
@st.cache_data(ttl=3600)
def predict_station_heights(station_code, from_date: str, to_date: str):
    # X_sim = pd.DataFrame([[hauteur_sim, heure_sim, mois_sim]], columns=["resultat_obs", "hour", "month"])
    # predictions = int(model.predict(X_sim)[0])

    url = f"{API_BASE_URL}/station/hixnj/predict"
    response = requests.post(url, json={"station_code": station_code, "from_date": from_date, "to_date": to_date})
    response.raise_for_status()
    content = response.json()
    # DEBUG : print("****** content:", content)
    assert(isinstance(content, dict))
    assert(all(field in content.keys() for field in ["api_version", "count", "predictions"]))
    predictions = content["predictions"]
    # DEBUG : print("****** height predictions:", predictions)
    assert(isinstance(predictions, list))
    assert((len(predictions) == 0) or (isinstance(predictions[0], dict) and (all(field in predictions[0].keys() for field in ["ds", "yhat"]))))
    return predictions

# OBSOLETE :
# @st.cache_data
# def get_risk(station_code, date: str):
#     predictions = predict_station_heights(station_code, date, date)
#     prediction = predictions[-1]  # get the last obs (of 1)
#     yhat = prediction["yhat"]

#     # TODO : use thresholds[q98]

#     # TMP :
#     match yhat:
#         case y if y < 2000.0:
#             level = 1
#         case _:
#             level = 3
#     return level, yhat


# ----------------------------------------------------------

st.set_page_config(
    page_title="Prévisions des Risques d'Inondation",
    page_icon="💦",
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

st.markdown("# Prévision des Risques de Crues Fluviales")
st.markdown("## Bassin versant de l'Ill - Grand Est")
st.markdown("---")

risk_colors = {1: "#28a745", 2: "#ffc107", 3: "#dc3545"}
risk_labels = {1: "🟢 FAIBLE", 2: "🟡 MODÉRÉ", 3: "🔴 ÉLEVÉ"}
risk_css    = {1: "risk-green", 2: "risk-yellow", 3: "risk-red"}

col1, col2 = st.columns([3, 2])

stations = load_stations()

with col1:
    st.markdown("### Carte des stations de prévision des crues")
    noms, lats, lons, couleurs, textes = [], [], [], [], []
    for station in stations:
        assert(isinstance(station, dict))
        station_site = station["site"]
        station_code = station["code"]
        station_label = station["label"]
        station_latitude = station["latitude"]
        station_longitude = station["longitude"]

        # date = dt.date.today().strftime("%Y-%m-%d")        
        # level, height = get_risk(station_code, date=date)
        
        noms.append(station_label)
        lats.append(station_latitude)
        lons.append(station_longitude)
        couleurs.append(risk_colors.get(1, "#888888"))
        textes.append(f"{station_label}<br>Site: {station_site}<br>Code: {station_code}")

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
        mapbox=dict(style="open-street-map", center=dict(lat=station_latitude, lon=station_longitude), zoom=32),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420
    )
    st.plotly_chart(fig_map, width='stretch')

# with col2:
#     st.subheader("🚨 Niveaux de vigilance")
#     for station in stations:
#         assert(isinstance(station, dict))
#         station_code = station["code"]
#         station_label = station["label"]
#         station_latitude = station["latitude"]
#         station_longitude = station["longitude"]

#         date = dt.date.today().strftime("%Y-%m-%d")        
#         level, height = get_risk(station_code, date=date)
#         css = risk_css.get(level, "risk-green")
#         label = risk_labels.get(level, "N/A")
#         st.markdown(f"""
#         <div class='risk-card {css}'>
#             {station_label}
#             <br>
#             <span style='font-size:1.2em'>{label}</span><br>
#             <small style='font-weight:normal'>Hauteur actuelle: {height:.0f} mm</small>
#         </div>
#         """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Observation des hauteurs d'eau par station")

station_code_by_labels = { station["label"]: station for station in stations }
station_labels = list(station_code_by_labels.keys())

default_station_code = "A236003001" # kogenheim
default_station_label = [s["label"] for s in stations if s["code"] == default_station_code][0]
default_station_label_index = [n for n, s in enumerate(station_labels) if s == default_station_label][0]

selected_name = st.selectbox("Sélectionner une station", options=station_labels, index=default_station_label_index)
selected_code = station_code_by_labels[selected_name]["code"]

station_dates, station_stats = load_station_dates_and_stats(selected_code)
station["dates"] = station_dates
station["stats"] = station_stats

# TODO : API/station/hixnj/dates(station_code) -> dates (min,max)

selected_date_1 = station_dates["lower"]
selected_date_2 = station_dates["upper"]

# TODO : API/station/hixnj/thresholds(station_code) -> thresholdstes (min,max,q98)
selected_threshold = station_stats["q98"]

quantity_label = "hauteur d'eau maximale journalière (HIXnJ)"
quantity_legend = "HIXnJ (mm)"

# key_h = f"{selected_code}_H"
# df_plot = dfs[key_h] if key_h in dfs else None
# if df_plot:
#     yobs = df_plot["resultat_obs"]
#     ds = df_plot["date_obs"]

selected_values = load_station_heights(selected_code, selected_date_1, selected_date_2)

fig_ts = go.Figure()
df_obss = pd.DataFrame(selected_values)
# DEBUG print("****** df_obss:", df_obss)
ds_obs = df_obss["ds"] if "ds" in df_obss.keys() else None
y_obs = df_obss["yobs"] if "yobs" in df_obss.keys() else None
if ds_obs is not None and y_obs is not None:
    fig_ts.add_trace(
        go.Scatter(
            x=ds_obs, 
            y=y_obs,
            mode="lines", 
            name=quantity_label,
            line=dict(color="#003189", width=2)
    ))

fig_ts.add_hline(
    y=selected_threshold, 
    line_dash="dash", 
    line_color="#dc3545", 
    annotation_text="Seuil d'alerte"
)
fig_ts.update_layout(
    title=f"Station de {selected_name} : {quantity_label}",
    xaxis_title="Date",
    yaxis_title=quantity_legend,
    height=350,
    # plot_bgcolor='white',
    # paper_bgcolor='white',
)
st.plotly_chart(fig_ts, width='stretch')

# --- Prédiction

# st.subheader("Prédictions des hauteurs d'eau")

# col3, col4, col5 = st.columns(3)
# with col3:
#     hauteur_sim = st.slider("Hauteur d'eau (mm)", 400, 1500, 800)
# with col4:
#     heure_sim = st.slider("Heure", 0, 23, 12)
# with col5:
#     mois_sim = st.slider("Mois", 1, 12, 6)

prediction_label = "Prédiction des hauteurs d'eau maximales journalières"

default_prediction_delta = dt.timedelta(days=30)

prediction_date_1 = selected_date_1
prediction_date_2 = "2026-06-10"

predictions = predict_station_heights(selected_code, prediction_date_1, prediction_date_2)

fig_ts = go.Figure()
df_preds = pd.DataFrame(predictions)
# DEBUG : print("****** df_preds:", df_preds)
ds_pred = df_preds["ds"] if "ds" in df_preds.keys() else None
# DEBUG : print("****** ds:", type(ds), len(ds))
y_pred = df_preds["yhat"] if "yhat" in df_preds.keys() else None
# DEBUG : print("****** yhat:", type(yhat), len(yhat))
if ds_pred is not None and y_pred is not None:
    if ds_obs is not None and y_obs is not None:
        fig_ts.add_trace(
            go.Scatter(
                x=ds_obs, 
                y=y_obs,
                mode="lines", 
                name="observations",
                line=dict(color="#003189", width=2)
        ))

    fig_ts.add_trace(
        go.Scatter(
            x=ds_pred, 
            y=y_pred,
            mode="lines", 
            name="prédictions",
            line=dict(dash="dash", color="#28a745", width=2)
    ))

fig_ts.add_hline(
    y=selected_threshold, 
    line_dash="dash", 
    line_color="#dc3545", 
    annotation_text="Seuil d'alerte"
)
fig_ts.update_layout(
    title=f"{prediction_label}",
    xaxis_title="Date",
    yaxis_title=quantity_legend,
    height=350,
)
st.plotly_chart(fig_ts, width='stretch')

# date = ds_pred.max()
# # DEBUG : print(f"****** date: {date} ({type(date)})")    
# level, _ = get_risk(station_code, date=date)
# css = risk_css.get(level, "risk-green")
# label = risk_labels.get(level, "N/A")

# st.markdown(f"""
# <div class='risk-card {css}' style='font-size:2em; padding:30px'>
#     Niveau de risque prédit : {label}
# </div>
# """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Machine Learning for Flood Forecasting - Jedha Data Science Fullstack Bootcamp 2026 - Data source: https://hubeau.eaufrance.fr")

# ==============================================================================
# LOCAL TEST

if __name__ == "__main__":  
    import subprocess
    import sys

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "4000",
        "--server.headless", "true"  # pas d'ouverture automatique du navigateur
    ])