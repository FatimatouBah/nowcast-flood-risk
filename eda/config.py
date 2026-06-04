# Local data directory
## Local data will be repackaed in ccsv files and store on the project's S3 bucket.

import os
LOCAL_DATA_DIRECTORY = os.path.abspath(os.path.join("data", "hubeau"))

# Hub'Eau's API's URL for hydrometric observations
HUBEAU_BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"

# Dates interval of interest
## Defined by stations that have "obs_elab" qualified data until 2026-06-01

DATA_TIME_PERIOD = ("2007-01-01", "2026-06-01")

# Sites de mesures hydrométries pour l'entraînement du modèle de prévision de risque d'inondation

SITES = {
    
    # Site principal (Kogenheim)
    
    "Kogenheim":  {
        "code" : "A2360030",
        "site": "main",
        "river": "ill",
        "region": "grand-est",
		"stations": [
			{"code": "A236003001", "municipality" :"kogenheim"}
		]
	},
    "Colmar-1":     {
        "code" : "A1580201", 
        "site": "main",
        "river": "launch",
        "region": "grand-est",
		"stations": [
            {"code": "A158020101", "municipality" :"colmar"}, 
        ]
	},
    "Colmar-2":     {
        "code" : "A1610030", 
        "site": "main",
        "river": "ill",
        "region": "grand-est",
		"stations": [
            {"code": "A161003001", "municipality" :"colmar"}, 
        ]
	},
    "Colmar-3":     {
        "code" : "A2220001", 
        "site": "main",
        "river": "fetch",
        "region": "grand-est",
		"stations": [
            {"code": "A222000101", "municipality" :"colmar"}, 
        ]
	},
    "Selestat":   {
        "code" : "A2350200",
        "site": "main",
        "river": "giessen",
        "region": "grand-est",
		"stations": [
			{"code": "A235020001", "municipality" :"selestat"},
			{"code": "A235020002", "municipality" :"selestat"},
			{"code": "A235020003", "municipality" :"selestat"}
		], 
	},
    "Ostheim":    {
        "code" : "A2140100",
        "site": "main",
        "river": "fetch",
        "region": "grand-est",
		"stations": [
			{"code": "A214010001", "municipality" :"ostheim"}
		],
	},

    # Site secondaire (Waltenheim)
    
    "Waltenheim": {
        "code" : "A3480200",
        "site": "secondary",
        "river": "zorn",
        "region": "grand-est",
		"stations": [
			{"code": "A348020001", "municipality" :"waltenheim-sur-zorn"}
		], 
	},
    "Oberhof":    {
        "code" : "A3430210",
        "site": "secondary",
        "river": "zinsel-sud",
        "region": "grand-est",
		"stations": [
			{"code": "A343021001", "municipality" :"eckartswiller"}
		], 
	},
    "Saverne":    { 
        "code" : "A3410200",
        "site": "secondary",
        "river": "zorn",
        "region": "grand-est",
		"stations": [
			{"code": "A341020001", "municipality" :"saverne"}
		]
	},
}