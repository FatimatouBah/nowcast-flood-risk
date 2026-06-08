# Local data directory
## Local data will be repackaed in ccsv files and store on the project's S3 bucket.

import os
LOCAL_DATA_DIRECTORY = os.path.abspath(os.path.join("..", "data", "hubeau"))

# Hub'Eau's API's URL for hydrometric observations
HUBEAU_BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"

# Dates interval of interest
## Defined by stations that have "obs_elab" qualified data until 2026-06-01

DATA_TIME_PERIOD = ("2007-01-01", "2026-06-01")

MAX_TRAINING_DATA_DATE = "2025-12-31"
CONFIGURATION_DATA_DATE_FORMAT = "%Y-%m-%d"

# Sites de mesures hydrométries pour l'entraînement du modèle de prévision de risque d'inondation

import sites
SITES = sites.SITES
