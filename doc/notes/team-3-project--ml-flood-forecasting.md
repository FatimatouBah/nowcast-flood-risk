Exploring machine learning methods for flood forecasting
===
Jedha DSFSFT41 Team 3 Project
---

# 0. Project information 

- Project Title:
    - Exploration des méthodes d'apprentissage automatique pour la prévision des crues (Exploring machine learning methods for flood forecasting)

- Field:
    - Gestion des risques naturels

- Team members:
    - Fatimatou Bah (en distanciel)
    - Didier Zef (en distanciel)
    - Nicolas Pichon (en distanciel)

- Dataset Links:
    - [Hub'Eau](https://hubeau.eaufrance.fr/page/api-hydrometrie)
    - [DBHI](https://www.georisques.gouv.fr/base-de-donnees/BDHI)
    - [open-meteo](https://open-meteo.com/)
    - [donneespubliques.meteofrance](https://donneespubliques.meteofrance.fr/)
    - [meteostat](https://dev.meteostat.net/)
    
## 1. Problématique

[...] 
[voir introduction de [ML4FF1] et [ML4FF2]]

###### Livrable
[...] [un système de nowcast du risque d'inondation qui combine précipitations récentes et prévues (CHIRPS), modèle numérique de terrain (SRTM), occupation des sols et historiques d'événements pour classifier le risque par cellule géographique, avec une interface cartographique et un curseur temporel.]

###### Impact attendu
[...]

## 2. Contour du livrable

- notebooks d'exploration et d'analyse des données
- pipeline d'entrainemnt mlflow
- pipeline de mise en production huggingface
- application web streamlit 
    - mvp = sélection site & dates + visualisation de la hauteur d'eau sur la période connue + prevision de la hauteur d'eau sur la période future
    - baseline du modèle de prédiction : Prophet

## 3. Sources des données

| Source                                                       | Type                                                                 | Accès                | Licence     |
|--------------------------------------------------------------|----------------------------------------------------------------------|----------------------|-------------|
| [Hub'Eau](https://hubeau.eaufrance.fr/page/api-hydrometrie)  | Chroniques des hauteurs et des débits d'eau des fleuves en france    | API libre            | ?           |
| [DBHI](https://www.georisques.gouv.fr/base-de-donnees/BDHI)  | Base de données historique des inondations en France                 | Téléchargement libre | Open Data ? |
| [open-meteo](https://open-meteo.com/)                        | Occupation des sols (imperméabilisation, forêt, agriculture, urbain) | API + téléchargement | Copernicus |
| [donneespubliques.meteofrance](https://donneespubliques.meteofrance.fr/) | Base de données historique des inondations en France | Téléchargement CSV | Open Data |
| [meteostat](https://dev.meteostat.net/) | Historique des surfaces en eau permanentes et saisonnières | Google Earth Engine | CC BY |

**Volumétrie** : zone d'étude = 1 bassin versant françai (l'Ill), 18,5 ans de chroniques (mesures de hauteurs et de débits d'eau).

**Conformité RGPD** : données publiques sans contrainte de traitement.

## 4. Architecture technique

## 5. Stack technique

## 6. Métriques de succès

## 7. Plannification

- **Budget** : 70H / personne (210H total)

#### Data Engineer
- [...]
- livrables?

#### ML Engineer
- [...]
- livrables?

#### MLOps Engineer
- [...]
- livrables?

#### Product Owner
- [...]
- livrables?

## x. Références
- [ML4FF1]
- [ML4FF2]