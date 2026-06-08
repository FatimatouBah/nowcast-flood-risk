Nowcast flood risk project
===

# J0
- articles sources & autres documentation --> TODO
- sources de données --> vu
    - hydroportial, hubeau, bdhi (frozen)
    - meteostat, meteofrance, open-meteo
- github sources --> DONE
- github project --> DONE (TODO : exploiter)

# J1
- recuperation des données --> TODO
- analyse/nettoyage des données --> TODO
- visualisation des données --> TODO

# Jx
- concevoir dashboard mvp (ex: https://vilaine-amont.haruni.net/) --> TODO
- mettre en place un environnement commun-> conda? venv? --> TODO

# MVP
- pipeline minimal mlflow + fastapi -> streamlit / huggingface
- doc minimale pour expliquer problematique et formulation
- dashboard minimal pour visualiser les données H/Q/Pluies vs. dates & site
    - ajouter prédictions selon modèle ML (permettre de visualiser la prédiction sur les périodes passées)
    - éventuellement, ajouter évènements crues, etc.
- baseline de prediction entrainé (linear regression) + métriques d'évaluation
    - eventuellement, modeles rf, gb, xgb + métriques d'évaluation + comparaison des modèles