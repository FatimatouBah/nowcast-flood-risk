import requests
import pandas as pd
import os

def fetch_hauteurs_ill(code_station='A161003001', size=1000):
    """Récupère les hauteurs d'eau de l'Ill à Colmar depuis Hub'eau"""
    
    url = 'https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr'
    params = {
        'code_entite': code_station,
        'grandeur_hydro': 'H',
        'size': size
    }
    
    r = requests.get(url, params=params)
    print(f"Statut API : {r.status_code}")
    
    data = r.json()
    print(f"Nombre total d'observations : {data['count']}")
    
    df = pd.DataFrame(data['data'])
    print(f"Colonnes : {df.columns.tolist()}")
    print(df.head())
    
    # Sauvegarder
    os.makedirs('data/hubeau', exist_ok=True)
    df.to_csv('data/hubeau/ill_colmar_hauteurs.csv', index=False)
    print("Données sauvegardées dans data/hubeau/ill_colmar_hauteurs.csv")
    
    return df