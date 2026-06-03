# Sites de mesures hydrométries pour l'entraînement du modèle de prévision de risque d'inondation
import os
LOCAL_DATA_DIRECTORY = os.path.abspath(os.path.join("data", "hubeau"))

TRAINING_TIME_PERIOD = ("2007-01-01", "2025-12-31")
TESTING_TIME_PERIOD = ("2026-01-01", "2026-06-01")