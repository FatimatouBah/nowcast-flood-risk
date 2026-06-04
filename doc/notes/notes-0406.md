# 0. 
...

# 1. pipeline d'apprentissage
- données CSV -> S3 -> clés d'accès -> .env local pour les tests locaux
- definir un environnement -> venv (requirements.txt) ou conda (conda.yaml)
- train.py -> sauve le modèle sur mlflow

# 2. pipeline de production backend
- un service d'API pour faire la prédiction -> FastAPI + Docker

# 3. pipeline de production frontend
- app.py qui appelle l'API, qui récupère des données ou des résultats et qui les visualise
- on a un docker encapsule app.py et qui est deployé sur HuggingFace
        - HF = réutiliser les comptes individuels (pour le MVP)

