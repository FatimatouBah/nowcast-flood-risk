# nowcast-flood-risk

**Système de nowcast du risque d'inondation par ML**

# Nowcast Risque d'Inondation 🌊

Système de prédiction en quasi-temps réel du risque d'inondation
à l'échelle d'un bassin versant, combinant données pluviométriques,
topographiques et d'occupation des sols.

**Zone d'étude** : Loire moyenne (Tours / Blois)
**Période** : 2000 – 2023

---

## Problématique

Les collectivités locales manquent d'outils accessibles pour anticiper
les crues en temps quasi-réel. Ce projet propose une approche ML légère,
opérationnelle et déployable sans infrastructure lourde.

---

## Stack technique

- Python · pandas · scikit-learn · XGBoost
- rasterio · geopandas · pysheds
- Streamlit · FastAPI · Folium
- MLflow · Docker

---

## Sources de données

| Source                | Contenu                           |
| --------------------- | --------------------------------- |
| CHIRPS (UCSB)         | Précipitations quotidiennes      |
| SRTM (NASA)           | Modèle numérique de terrain 30m |
| Copernicus Land Cover | Occupation des sols               |
| BDHI (data.gouv.fr)   | Historique des inondations France |

---

## Lancer le projet

```bash
git clone https://github.com/FatimatouBah/nowcast-flood-risk.git
cd nowcast-flood-risk
pip install -r requirements.txt
```

---

## Équipe

Projet réalisé dans le cadre du Mastère Architecte IA — Jedha Bootcamp
