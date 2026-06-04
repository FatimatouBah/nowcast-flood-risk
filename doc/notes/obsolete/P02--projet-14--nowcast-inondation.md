# Projet 14 — Nowcast Risque d'Inondation
### Prédire le risque d'inondation local en quasi-temps réel (nowcast) sur la base de données pluviométrique, topographiques et d'occupation des sols
*Certification RNCP 35288 — Concepteur Développeur en Science des Données*

---

## 1. Problématique métier

Les services de gestion des crises, collectivités et protection civile manquent d'outils locaux et accessibles pour anticiper en temps quasi-réel le risque d'inondation à l'échelle d'un bassin versant ou d'une commune. Les modèles hydrologiques classiques (HEC-RAS, MIKE) sont coûteux, complexes à calibrer et réservés aux grandes structures. Une approche ML légère combinant données pluviométriques, topographiques et d'occupation des sols permet de produire une estimation du risque opérationnelle et accessible.

**Valeur délivrée** : un système de nowcast du risque d'inondation qui combine précipitations récentes et prévues (CHIRPS), modèle numérique de terrain (SRTM), occupation des sols et historiques d'événements pour classifier le risque par cellule géographique, avec une interface cartographique et un curseur temporel.

**Impact estimé** : selon la Banque Mondiale, les inondations représentent 40% des catastrophes naturelles mondiales. Un système de nowcast accessible aux collectivités locales permet d'anticiper les évacuations et de réduire les dommages humains et matériels. En France, le coût annuel moyen des inondations dépasse 1 milliard d'euros.

---

## 2. Données

| Source | Type | Accès | Licence |
|--------|------|-------|---------|
| CHIRPS (UCSB) | Précipitations quotidiennes historiques et quasi-temps-réel, résolution 0.05° | Téléchargement gratuit | Domaine public |
| SRTM (NASA) | Modèle numérique de terrain 30m — altitude, pente, direction d'écoulement | Téléchargement gratuit | Domaine public |
| Copernicus Land Cover | Occupation des sols (imperméabilisation, forêt, agriculture, urbain) | API + téléchargement | Copernicus |
| BDHI (data.gouv.fr) | Base de données historique des inondations en France | Téléchargement CSV | Open Data |
| JRC Global Surface Water | Historique des surfaces en eau permanentes et saisonnières | Google Earth Engine | CC BY |
| OpenStreetMap | Réseau hydrographique, infrastructures | API Overpass | ODbL |

**Volume estimé** : zone d'étude de 1 à 3 bassins versants français, 10 à 30 ans d'historique pluviométrique → ~500 000 à 2 000 000 lignes selon la résolution spatiale.

**Conformité RGPD** : données environnementales, topographiques et historiques entièrement publiques. Aucune donnée personnelle traitée. Aucune contrainte de traitement.

---

## 3. Architecture technique

```
[CHIRPS pluie] ────┐
[SRTM DEM] ────────┤
[Copernicus LC] ───┼──► [Pipeline ETL géospatial] ──► [Feature Engineering]
[JRC surface eau] ─┤         │                               │
[OSM hydro] ───────┤    (rasterio, geopandas,        (dérivés topographiques :
[BDHI historiques] ┘     nettoyage, grille)            pente, TWI, distance cours
                                                        d'eau, imperméabilisation,
                                                        indice humidité antécédente)
                                                               │
                                                    [Génération dataset]
                                                    (cellules grille × temps,
                                                     label 0/1 inondation)
                                                               │
                                              ┌────────────────┤
                                              │                │
                                      [XGBoost / RF]    [Baseline logistique]
                                      (classification   (référence simple)
                                       risque 3 niveaux)
                                              │
                                      [Évaluation]
                                  (AUC, F1, spatial CV,
                                   calibration probabiliste)
                                              │
                                      [MLflow tracking]
                                              │
                                   [API FastAPI /risk]
                                   (requête coordonnées
                                    → score risque J+1/J+3)
                                              │
                                  [Dashboard Streamlit]
                                  (carte choroplèthe + curseur
                                   temporel + alertes)
                                              │
                              [Docker + Streamlit Cloud]
```

---

## 4. Stack technique

| Composant | Outil | Justification |
|-----------|-------|---------------|
| Collecte pluie | `requests` + CHIRPS FTP/HTTP | Données quasi-temps-réel disponibles sous 2 jours |
| Géospatial | `rasterio`, `geopandas`, `shapely` | Standard Python rasters et vecteurs |
| Dérivés topographiques | `richdem`, `pysheds` | TWI, pente, direction écoulement depuis DEM |
| Feature engineering | `pandas`, `numpy`, `scikit-learn` | Fenêtrage temporel, agrégations spatiales |
| Modélisation | `XGBoost`, `LightGBM`, `scikit-learn` (RF) | Benchmark classification rapide |
| Calibration | `sklearn.calibration` (Platt, isotonic) | Probabilités calibrées = crédibilité des alertes |
| Évaluation | AUC-ROC, F1, Brier Score, spatial cross-validation | Split spatial obligatoire (voir note) |
| Explainabilité | `shap` | Facteurs déterminants par cellule |
| Cartographie | `Folium`, `Plotly` | Cartes interactives choroplèthes |
| Interface | `Streamlit` | Curseur temporel, carte risque, alertes couleur |
| Tracking | `MLflow` | Log des runs, comparaison modèles |
| Déploiement | `Docker` + Streamlit Cloud | Gratuit, public, stable pour démo |

---

## 5. Métriques de succès

| Métrique | Objectif |
|----------|----------|
| AUC-ROC (split spatial) | ≥ 0.80 |
| F1-score (classe inondation) | ≥ 0.70 |
| Brier Score (calibration) | ≤ 0.15 |
| Comparaison modèles | Au moins 3 modèles benchmarkés |
| Dashboard opérationnel | Curseur J+1 à J+3 fonctionnel en démo |
| Temps de réponse API | < 3 secondes par requête |

> **Note méthodologique** : utiliser un **split temporel ET spatial** pour l'évaluation (entraînement sur années N-3 à N-1, test sur année N, zones géographiques distinctes). Un split aléatoire sur des données spatio-temporelles surestime massivement les performances réelles.

---

## 6. Version 1 personne — 80 heures

### Planning

| Semaine | Phase | Tâches | Heures |
|---------|-------|--------|--------|
| S1 | Cadrage & collecte | Sélection bassin versant FR, téléchargement CHIRPS + SRTM + Copernicus LC, exploration | 12h |
| S1-S2 | Feature engineering | Dérivés topographiques (pente, TWI, distance cours d'eau), indice humidité antécédente, imperméabilisation | 14h |
| S2 | Génération dataset | Grille spatiale, labellisation depuis BDHI, split spatial/temporel | 8h |
| S2-S3 | Modélisation | Baseline logistique → Random Forest → XGBoost, comparaison AUC/F1 | 14h |
| S3 | Évaluation & SHAP | Métriques par niveau de risque, SHAP top features, calibration probabiliste | 8h |
| S4 | Interface | Streamlit : carte choroplèthe risque + curseur temporel J+1/J+3 + alertes couleur | 10h |
| S4 | Déploiement | Dockerfile, push Streamlit Cloud | 6h |
| S5 | Soutenance | Slides, démo, impact gestion crise, RGPD | 8h |
| **Total** | | | **80h** |

### Périmètre MVP

- 1 bassin versant français (ex : Garonne aval, Rhône moyen, Loire) — zone connue pour ses crues
- Variables : CHIRPS (pluie 5 derniers jours + prévision J+1), altitude, pente, TWI, distance cours d'eau, occupation des sols
- Modèles : baseline logistique + Random Forest + XGBoost
- Risque en 3 niveaux : faible / modéré / élevé
- Interface : carte + curseur temporel J+1 à J+3

> **Conseil** : la qualité des labels (événements historiques BDHI) est la principale source d'incertitude. Vérifier manuellement un échantillon des événements recensés avant d'entraîner le modèle.

---

## 7. Version 3 personnes — 80h chacun (240h total)

### Répartition des rôles

#### Personne A — Data Engineer (80h)

| Tâches | Heures |
|--------|--------|
| Pipeline multi-sources : CHIRPS (pluie) + SRTM + Copernicus LC + JRC surface water | 14h |
| Calcul dérivés topographiques : pente, aspect, TWI, direction écoulement (`richdem`, `pysheds`) | 14h |
| Alignement spatial des rasters (CRS, résolution, extent) sur grille commune | 10h |
| Collecte et nettoyage historiques d'inondation BDHI + événements COPERNICUS EMS | 10h |
| Scheduler de mise à jour quotidienne CHIRPS (données quasi-temps-réel) | 8h |
| Documentation pipeline + tests de reproductibilité géospatiale | 8h |
| Contribution soutenance | 8h |
| Buffer | 8h |
| **Total** | **80h** |

#### Personne B — ML Engineer (80h)

| Tâches | Heures |
|--------|--------|
| Benchmark : logistique vs Random Forest vs XGBoost vs LightGBM | 14h |
| Feature engineering avancé : indice humidité antécédente (API), SPI (Standardized Precipitation Index) | 10h |
| Évaluation rigoureuse : split spatio-temporel, AUC / F1 / Brier Score | 10h |
| Calibration probabiliste (Platt scaling, isotonic regression) | 8h |
| Explainabilité SHAP : top features par zone géographique et par saison | 10h |
| Extension multi-bassins : 3 bassins versants français comparés | 10h |
| Rapport de modélisation comparatif | 8h |
| Contribution soutenance | 8h |
| Buffer | 2h |
| **Total** | **80h** |

#### Personne C — MLOps & Product (80h)

| Tâches | Heures |
|--------|--------|
| API FastAPI : `/risk/{lat}/{lon}` → score risque + facteurs contributeurs | 12h |
| Dashboard Streamlit avancé : carte choroplèthe Folium, curseur temporel J+1→J+3, alertes couleur, SHAP par zone | 16h |
| Système d'alertes automatiques si risque élevé détecté sur zone définie | 8h |
| MLflow : tracking expériences, registre de modèles versionnés | 8h |
| Docker multi-services + CI/CD GitHub Actions → Streamlit Cloud | 10h |
| Tests end-to-end (pipeline collecte → prédiction → affichage) | 8h |
| Contribution soutenance | 8h |
| Buffer | 10h |
| **Total** | **80h** |

### Périmètre étendu (version équipe)

- 3 bassins versants français comparés (Garonne, Rhône, Loire)
- 6 variables explicatives supplémentaires : SPI, indice humidité antécédente, densité de réseau hydrographique OSM
- Calibration probabiliste documentée avec courbes de fiabilité
- Système d'alertes automatiques par zone
- Explainabilité SHAP géographiquement localisée ("la zone X est à risque élevé car pente forte + sol imperméable + pluie cumulée > 80mm")

---

## 8. Version 2 personnes — 80h chacun (160h total)

### Principe de répartition

Personne A gère tout le pipeline géospatial (collecte, dérivés topographiques, génération du dataset). Personne B prend en charge la modélisation, l'évaluation, l'interface et le déploiement.

### Répartition des rôles

#### Personne A — Data & Géospatial (80h)

| Tâches | Heures |
|--------|--------|
| Pipeline collecte : CHIRPS + SRTM + Copernicus Land Cover | 14h |
| Calcul dérivés topographiques : pente, TWI, direction écoulement (`richdem`, `pysheds`) | 14h |
| Alignement spatial des rasters (CRS, résolution, extent) sur grille commune | 8h |
| Collecte et nettoyage historiques inondations BDHI | 8h |
| Génération dataset : grille spatiale, labellisation, split spatio-temporel | 10h |
| Documentation pipeline géospatial + tests | 10h |
| Contribution soutenance (slides ETL + architecture géospatiale) | 8h |
| Buffer | 8h |
| **Total** | **80h** |

#### Personne B — ML & Product (80h)

| Tâches | Heures |
|--------|--------|
| Benchmark : baseline logistique + Random Forest + XGBoost — AUC/F1/Brier | 16h |
| Calibration probabiliste (Platt scaling) + courbes de fiabilité | 8h |
| Explainabilité SHAP : top features par zone et par saison | 8h |
| Dashboard Streamlit : carte choroplèthe Folium, curseur temporel J+1→J+3, alertes couleur | 16h |
| API FastAPI : `/risk/{lat}/{lon}` → score + facteurs contributeurs | 10h |
| MLflow tracking + Docker + CI/CD → Streamlit Cloud | 10h |
| Contribution soutenance (démo live + slides ML + calibration) | 8h |
| Buffer | 4h |
| **Total** | **80h** |

### Périmètre version 2 personnes

- 1 à 2 bassins versants français (vs 1 solo / 3 en équipe de 3)
- Variables : CHIRPS + pente + TWI + distance cours d'eau + imperméabilisation
- Benchmark 3 modèles (logistique, RF, XGBoost) avec calibration probabiliste
- SHAP par zone
- Curseur temporel J+1→J+3 + alertes couleur
- Docker + CI/CD

---

## 9. Ce qui fait la différence

1. **Split spatio-temporel documenté et justifié** — expliquer pourquoi un split aléatoire biaiserait les performances sur des données géospatiales corrélées
2. **Calibration probabiliste** — un score de risque crédible (0.72 = 72% de probabilité d'inondation) vaut mieux qu'une simple classification binaire pour les décideurs
3. **SHAP géographiquement localisé** — "ce pixel est à risque élevé car imperméabilisation 89% + pluie cumulée 5j > seuil + pente aval forte"
4. **Curseur temporel J+1→J+3 en démo live** — visualisation immédiatement compréhensible par un public non-technique
5. **Benchmark calibré 3-4 modèles** avec tableau AUC/F1/Brier comparatif — montre la rigueur scientifique
6. **Impact chiffré** : "une alerte 48h à l'avance permet aux collectivités d'activer les plans de prévention et de réduire les dommages estimés de X%"

---

## 10. Références & ressources

- CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) : https://www.chc.ucsb.edu/data/chirps
- SRTM DEM (NASA) : https://www2.jpl.nasa.gov/srtm/
- Copernicus Land Cover : https://land.copernicus.eu/pan-european/corine-land-cover
- JRC Global Surface Water : https://global-surface-water.appspot.com/
- BDHI (Base de Données Historique des Inondations) : https://www.georisques.gouv.fr/
- pysheds (hydrologie Python) : https://mattbartos.com/pysheds/
- richdem (dérivés topographiques) : https://richdem.readthedocs.io/
- Copernicus Emergency Management Service (événements) : https://emergency.copernicus.eu/
- Brier Score et calibration : https://scikit-learn.org/stable/modules/calibration.html
