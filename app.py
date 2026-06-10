import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from datetime import datetime

# Une seule configuration au début
st.set_page_config(page_title="Nowcast Inondation", page_icon="💦", layout="wide")

st.markdown("""
<style>
    .risk-card { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin: 8px 0; }
    .risk-green  { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .risk-yellow { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .risk-red    { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    h1 { color: #003189; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    for root, _, files in os.walk("mlruns"):
        for f in files:
            if f == "model.pkl": return joblib.load(os.path.join(root, f))
    if os.path.exists("model.pkl"): return joblib.load("model.pkl")
    return None

model = load_model()

# Récupération automatique des noms de colonnes attendus par le modèle
# Si le modèle n'a pas cette info, on prend les noms classiques par défaut
expected_cols = getattr(model, "feature_names_in_", ["resultat_obs", "hour", "month"])

@st.cache_data
def load_local_data():
    st.write("Chargement des données en cours...") # Ajout pour le test
    dfs = {}
    path = "hubeau"
    if not os.path.exists(path):
        st.error(f"Le dossier {path} est introuvable !")
        return {}
    # ... le reste de ta fonction ...
    return dfs
stations = {
    "A161003001": {"nom": "Ill à Colmar", "lat": 48.080, "lon": 7.358},
    "A214010001": {"nom": "Fecht à Ostheim", "lat": 48.166, "lon": 7.358},
    "A236003001": {"nom": "Ill à Kogenheim", "lat": 48.266, "lon": 7.533},
    "A348020001": {"nom": "Zorn à Waltenheim", "lat": 48.716, "lon": 7.583},
}

dfs = load_data()

# --- MODIFICATION DE LA PARTIE PRÉDICTION ---
def get_risk(hauteur, heure, mois):
    if model is None: return 1
    
    # Création dynamique des données d'entrée
    data_dict = {
        "resultat_obs": hauteur,
        "hour": heure,
        "month": mois
    }
    
    # On construit le vecteur d'entrée selon l'ordre exact attendu par le modèle
    input_list = []
    for col in expected_cols:
        # On essaie de faire correspondre nos variables aux noms de colonnes du modèle
        if any(x in col.lower() for x in ["hauteur", "resultat"]): input_list.append(hauteur)
        elif any(x in col.lower() for x in ["hour", "heure"]): input_list.append(heure)
        elif any(x in col.lower() for x in ["month", "mois"]): input_list.append(mois)
        else: input_list.append(0) # Valeur par défaut si colonne inconnue
        
    X = pd.DataFrame([input_list], columns=expected_cols)
    try:
        pred = int(model.predict(X)[0])
        return max(1, min(pred, 3))
    except:
        return 1

# [Reste de l'interface identique à ton code précédent...]