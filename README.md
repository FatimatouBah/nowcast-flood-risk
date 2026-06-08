# nowcast-flood-risk

**Système de nowcast du risque d'inondation par ML**

[sources](https://github.com/FatimatouBah/nowcast-flood-risk/)

# Nowcast Risque d'Inondation 🌊

~~Système de prédiction en quasi-temps réel du risque d'inondation
à l'échelle d'un bassin versant, combinant données pluviométriques,
topographiques et d'occupation des sols.~~

**Zone d'étude** : Bassin versant de l'Ill — Grand Est (Alsace)

---

## Problématique

~~Les collectivités locales manquent d'outils accessibles pour anticiper
les crues en temps quasi-réel. Ce projet propose une approche ML légère,
opérationnelle et déployable sans infrastructure lourde.~~ 

---

## Architecture
- Explanatory Data Analysis -> compréhension des données Hub'Eau
- Prediction Model Training -> Etude d'un modèle de prédiction journalière des hauteurs d'eau sur le site de référence
- API Server -> API Web pour le dashboard
- Dasboard -> Application Web pour visualiser les hauteurs d'eau d'un site donné et les prédire  

---

## Stack technique

- EDA : Python, Pandas
- Training : Prophet, ...
- Serveur MLflow : MLflow, Huggingface, Docker
- Serveur API : FastAPI, Huggingface, Docker
- Dashboard Web : Streamlit, Huggingface, Docker

---

## Sources de données

- [hubeau](https://hubeau.eaufrance.fr/api/v2/hydrometrie)

## Références
- Évaluation des performances de l’intelligence artificielle  et de l’apprentissage automatique pour la prévision des crues : 
étude de cas du bassin versant de l’Ill 
    - N. REIMINGER1,2*, X. JURADO1, L. SAUNIER1, L. MAURER 2,3 , E. REIMINGER 4, L. WEBER1, T.H.L. NGUYEN 1,2,  C. WEMMERT2 
    - TSM numéro 11 - 2024 - 119e année
- Amélioration de la prévision des crues par l’utilisation de réseaux de neurones récurrents de type LSTM : 
étude de cas du bassin versant de l’Ill et de la Sarre
    - L. WEBER1, X. JURADO1, T.H.L. NGUYEN1,2, L. MAURER 2,3, E. REIMINGER 4, C. WEMMERT 2, N. REIMINGER 1,2
    - TSM numéro 9 - 2025
