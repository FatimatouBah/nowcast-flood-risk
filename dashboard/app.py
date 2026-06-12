import requests
from typing import Optional
import time
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------

API_BASE_URL = "https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space"

# wake-up the API server before calling any endpoint:
def _wake_up_api(timeout=60, interval=1):
    """Wait for the API Server to be ready before calling anything."""
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{API_BASE_URL}/ping", timeout=10)
                if r.status_code == 200 and r.json().get("status") is True:
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False
    except Exception:
        pass  # Silencieux, juste pour réveiller le Space

_wake_up_api()

# @st.cache_data
def load_stations():
    try:
        url = f"{API_BASE_URL}/stations"
        response = requests.get(url)
        response.raise_for_status()
    except Exception as exc:
        st.error(f"❌ Error requesting stations at '{url}': {exc}")
        return []
    else:       
        content = response.json()
        # DEBUG : print("****** content:", content)
        assert(isinstance(content, dict))
        assert(all(field in content.keys() for field in ["api_version", "count", "stations"]))
        stations = content["stations"]
        # DEBUG : print("****** stations:", stations)
        assert(isinstance(stations, list))
        assert((len(stations) == 0) or (isinstance(stations[0], dict) and (all(field in stations[0].keys() for field in ["site", "code", "label", "latitude", "longitude"]))))
        return stations

@st.cache_data
def load_station_dates_and_stats(station_code: str):
    try:
        url = f"{API_BASE_URL}/station/hixnj/stats"
        response = requests.get(url, params={"station_code": station_code})
        response.raise_for_status()
    except Exception as exc:
        st.error(f"❌ Error requesting dates and stats at '{url}' for station: '{station_code}': {exc}")
        return {}, {}
    else:
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

@st.cache_data
def load_station_heights(station_code: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
    try:
        url = f"{API_BASE_URL}/station/hixnj/observations"
        params = params={"station_code": station_code}
        if from_date is not None:
            params["from_date"] = from_date
        if to_date is not None:
            params["to_date"] = to_date
        response = requests.get(url, params=params)
        response.raise_for_status()
    except Exception as exc:
        st.error(f"❌ Error requesting heights at '{url}' for station: '{station_code}': {exc}")
        return []
    else:
        content = response.json()
        # DEBUG : print("****** content:", content)
        assert(isinstance(content, dict))
        assert(all(field in content.keys() for field in ["api_version", "count", "observations"]))
        observations = content["observations"]
        # DEBUG : print("****** height observations:", observations)
        assert(isinstance(observations, list))
        assert((len(observations) == 0) or (isinstance(observations[0], dict) and (all(field in observations[0].keys() for field in ["ds", "yobs"]))))
        return observations

@st.cache_data
def predict_station_heights(station_code, from_date: str, to_date: str):
    try:
        url = f"{API_BASE_URL}/station/hixnj/predict"
        response = requests.post(url, json={"station_code": station_code, "from_date": from_date, "to_date": to_date})
        response.raise_for_status()
    except Exception as exc:
        st.error(f"❌ Error predicting heights at '{url}' for station: '{station_code}': {exc}")
        return []
    else:
        content = response.json()
        # DEBUG : print("****** content:", content)
        assert(isinstance(content, dict))
        assert(all(field in content.keys() for field in ["api_version", "count", "predictions"]))
        predictions = content["predictions"]
        # DEBUG : print("****** height predictions:", predictions)
        assert(isinstance(predictions, list))
        assert((len(predictions) == 0) or (isinstance(predictions[0], dict) and (all(field in predictions[0].keys() for field in ["ds", "yhat"]))))
        return predictions

# ─── CONFIGURATION ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prédiction des Crues Fluviales en France",
    page_icon="💦",
    layout="wide"
)

# st.markdown("""
# <style>
#     .main { background-color: #f0f4f8; }
#     .risk-card {
#         padding: 20px;
#         border-radius: 12px;
#         text-align: center;
#         font-size: 1.5em;
#         font-weight: bold;
#         margin: 10px 0;
#     }
#     .risk-green  { background-color: #d4edda; color: #155724; border: 2px solid #28a745; }
#     .risk-yellow { background-color: #fff3cd; color: #856404; border: 2px solid #ffc107; }
#     .risk-red    { background-color: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
#     h1 { color: #003189; }
# </style>
# """, unsafe_allow_html=True)

# Style titre et le fond
st.markdown("""<style> h1, h2, h3 { color: #003189; }</style>
""", unsafe_allow_html=True)

# ─── INTERFACE ────────────────────────────────────────────────────────────────

st.title("💦 Prédiction des Crues Fluviales en France")
st.markdown("**Site d'étude : bassin versant de l'Ill (Grand Est)**")
st.markdown("---")

col1, col2 = st.columns([3, 2])

stations = load_stations()
if not stations:
    st.error(f"❌ Erreur au chargelentdes stations: aucune station n'a été trouvée")
    st.stop()

assert(len(stations) > 0)

# ─── CARTE ────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Stations de prévision des crues")
st.map(pd.DataFrame(stations), latitude="latitude", longitude="longitude", size=200, color="#02FF24")

st.markdown("---")

# with col1:
#     st.markdown("### Carte des stations de prévision des crues")
#     noms, lats, lons, couleurs, textes = [], [], [], [], []
#     for station in stations:
#         assert(isinstance(station, dict))
#         station_site = station["site"]
#         station_code = station["code"]
#         station_label = station["label"]
#         station_latitude = station["latitude"]
#         station_longitude = station["longitude"]
#         # date = dt.date.today().strftime("%Y-%m-%d")        
#         # level, height = get_risk(station_code, date=date)
#         noms.append(station_label)
#         lats.append(station_latitude)
#         lons.append(station_longitude)
#         couleurs.append(risk_colors.get(1, "#888888"))
#         textes.append(f"{station_label}<br>Site: {station_site}<br>Code: {station_code}")
#     fig_map = go.Figure(go.Scattermap(
#         lat=lats, lon=lons,
#         mode='markers+text',
#         marker=dict(size=20, color=couleurs, opacity=0.9),
#         text=noms,
#         textposition="top right",
#         hovertext=textes,
#         hoverinfo='text'
#     ))
#     fig_map.update_layout(
#         mapbox=dict(style="open-street-map", center=dict(lat=station_latitude, lon=station_longitude), zoom=32),
#         margin=dict(l=0, r=0, t=0, b=0),
#         height=420
#     )
#     st.plotly_chart(fig_map, width='stretch')

# ─── PARAMÈTRES + GRAPHIQUE ──────────────────────────────────────────────────

col_params, col_graph = st.columns([1, 3])

stations_by_names = { s["label"]: s for s in stations }
station_names = list(stations_by_names.keys())

default_station_code = "A236003001" # kogenheim
default_station_labels = [s["label"] for s in stations if s["code"] == default_station_code]
if (len(default_station_labels) == 0):
    st.error(f"❌ Error with stations: cannot find default station's code ({default_station_code})")
    st.stop()
default_station_label = default_station_labels[0]

default_station_label_indexes = [n for n, s in enumerate(station_names) if s == default_station_label]
if (len(default_station_label_indexes) == 0):
    st.error(f"❌ Error with stations: cannot find default station's name (Kogenheim)")
    st.stop()
default_station_label_index = default_station_label_indexes[0]

with col_params:
    st.subheader("⚙️ Paramètres")
    selected_name = st.selectbox("Choisir une station", options=station_names, index=default_station_label_index)
    selected_station = stations_by_names[selected_name]
    selected_code = selected_station["code"]

    station_dates, station_stats = load_station_dates_and_stats(selected_code)
    selected_station["dates"] = station_dates
    selected_station["stats"] = station_stats

    date_min = dt.date.fromisoformat(station_dates["lower"])
    date_max = dt.date.fromisoformat(station_dates["upper"])

    selected_date_1 = st.date_input("Date de la première observation", value=date_min, min_value=date_min, max_value=date_max)
    selected_date_2 = st.date_input("Date de la dernière observation", value=date_max, min_value=date_min, max_value=date_max)
    
    selected_threshold = station_stats["q98"]
    st.metric("Seuil d'alerte", f"{int(selected_threshold)} mm")

quantity_label = "hauteur d'eau maximale journalière (HIXnJ)"
quantity_legend = "HIXnJ (mm)"

# ds_obs = df_obss["ds"] if "ds" in df_obss.keys() else None
# y_obs = df_obss["yobs"] if "yobs" in df_obss.keys() else None
with col_graph:
    st.subheader(f"📈 Observation des hauteurs d'eau par station")
    selected_values = load_station_heights(selected_code, selected_date_1.isoformat(), selected_date_2.isoformat())
    df_obss = pd.DataFrame(selected_values)
    ds_obs = pd.to_datetime(df_obss["ds"]) if not df_obss.empty and "ds" in df_obss.columns else None
    y_obs = df_obss["yobs"] if not df_obss.empty and "yobs" in df_obss.columns else None

    fig = go.Figure()

    if ds_obs is not None and y_obs is not None:
        fig.add_trace(go.Scatter(
            x=ds_obs, y=y_obs, mode="lines", 
            name=quantity_label, 
            line=dict(color="#003189", width=2)
        ))

        # Seuil d'alerte
        fig.add_hline(y=selected_threshold, 
                      line_dash="dash", line_color="red", line_width=3,
                      annotation_text="Seuil d'alerte",
                      annotation_position="top left",
                      annotation_font_color="white")

    # Mise en forme graphique avec fond sombre pour lisibilité en présentation
    fig.update_layout(
        title=f"Station de {selected_name} : {quantity_label}",
        xaxis_title="Date", yaxis_title=quantity_legend, height=500,
        xaxis=dict(rangeslider=dict(visible=True))
    )
    st.plotly_chart(fig, use_container_width=True)

# *** 

# st.markdown("### Observation des hauteurs d'eau par station")

# station_code_by_labels = { station["label"]: station for station in stations }
# station_labels = list(station_code_by_labels.keys())

# default_station_code = "A236003001" # kogenheim
# default_station_label = [s["label"] for s in stations if s["code"] == default_station_code][0]
# default_station_label_index = [n for n, s in enumerate(station_labels) if s == default_station_label][0]

# selected_name = st.selectbox("Sélectionner une station", options=station_labels, index=default_station_label_index)
# selected_code = station_code_by_labels[selected_name]["code"]

# station_dates, station_stats = load_station_dates_and_stats(selected_code)
# station["dates"] = station_dates
# station["stats"] = station_stats

# # TODO : API/station/hixnj/dates(station_code) -> dates (min,max)

# selected_date_1 = station_dates["lower"]
# selected_date_2 = station_dates["upper"]

# # TODO : API/station/hixnj/thresholds(station_code) -> thresholdstes (min,max,q98)
# selected_threshold = station_stats["q98"]

# quantity_label = "hauteur d'eau maximale journalière (HIXnJ)"
# quantity_legend = "HIXnJ (mm)"

# selected_values = load_station_heights(selected_code, selected_date_1, selected_date_2)

# fig_ts = go.Figure()
# df_obss = pd.DataFrame(selected_values)
# # DEBUG print("****** df_obss:", df_obss)
# ds_obs = df_obss["ds"] if "ds" in df_obss.keys() else None
# y_obs = df_obss["yobs"] if "yobs" in df_obss.keys() else None
# if ds_obs is not None and y_obs is not None:
#     fig_ts.add_trace(
#         go.Scatter(
#             x=ds_obs, 
#             y=y_obs,
#             mode="lines", 
#             name=quantity_label,
#             line=dict(color="#003189", width=2)
#     ))

# fig_ts.add_hline(
#     y=selected_threshold, 
#     line_dash="dash", 
#     line_color="#dc3545", 
#     annotation_text="Seuil d'alerte"
# )
# fig_ts.update_layout(
#     title=f"Station de {selected_name} : {quantity_label}",
#     xaxis_title="Date",
#     yaxis_title=quantity_legend,
#     height=350,
#     # plot_bgcolor='white',
#     # paper_bgcolor='white',
# )
# st.plotly_chart(fig_ts, width='stretch')

# *** 

# --- Prédiction

PREDICTION_DAYS = 90

prediction_label = f"Prédiction des hauteurs d'eau maximales journalières sur {PREDICTION_DAYS} jours"

default_prediction_delta = dt.timedelta(days=PREDICTION_DAYS)

prediction_date_1 = selected_date_1
prediction_date_2 = selected_date_2 + default_prediction_delta

predictions = predict_station_heights(selected_code, prediction_date_1.isoformat(), prediction_date_2.isoformat())

fig_ts = go.Figure()
df_preds = pd.DataFrame(predictions)
# DEBUG : print("****** df_preds:", df_preds)
ds_pred = pd.to_datetime(df_preds["ds"]) if "ds" in df_preds.keys() else None
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
    height=500,
    xaxis=dict(rangeslider=dict(visible=True))
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