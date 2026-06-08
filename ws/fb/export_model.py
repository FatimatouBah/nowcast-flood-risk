import streamlit as st
import joblib
import numpy as np
import os

@st.cache_resource
def load_model():
    possible_paths = ["best_model.pkl", "src/best_model.pkl", "../best_model.pkl"]
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("Le fichier best_model.pkl est introuvable.")

model = load_model()

st.title("🌊 Prédiction du Risque d'Inondation")
st.write("Entrez les paramètres pour évaluer le risque d'inondation en temps réel.")

precipitations = st.slider("Précipitations (mm)", 0, 200, 50)
niveau_eau = st.slider("Niveau de la rivière (m)", 0.0, 10.0, 2.5)

if st.button("Calculer le risque"):
    features = np.array([[precipitations, niveau_eau]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("⚠️ Risque d'inondation ÉLEVÉ !")
    else:
        st.success("✅ Situation normale. Aucun risque détecté.")