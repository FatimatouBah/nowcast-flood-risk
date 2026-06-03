import requests
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_station_chunk(station_code, grandeur, date_debut, date_fin):
    """Fetch observations for a station between two dates"""
    url = 'https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr'
    all_data = []
    cursor = None

    while True:
        params = {
            'code_entite': station_code,
            'grandeur_hydro': grandeur,
            'date_debut_obs': date_debut,
            'date_fin_obs': date_fin,
            'size': 500
        }
        if cursor:
            params['cursor'] = cursor

        r = requests.get(url, params=params)

        if r.status_code not in [200, 206]:
            return []

        data = r.json()
        if not data.get('data'):
            break

        all_data.extend(data['data'])

        next_url = data.get('next')
        if not next_url or len(all_data) >= data['count']:
            break

        cursor = next_url.split('cursor=')[1].split('&')[0]

    return all_data

def fetch_water_levels(station_code, grandeur='H'):
    """Fetch all observations by chunking by month"""
    
    all_data = []
    start = datetime(2024, 1, 1)
    end = datetime.today()
    current = start

    while current < end:
        next_month = current + timedelta(days=30)
        date_debut = current.strftime('%Y-%m-%d')
        date_fin = min(next_month, end).strftime('%Y-%m-%d')

        print(f"Fetching {station_code} {grandeur} — {date_debut} to {date_fin}")
        chunk = fetch_station_chunk(station_code, grandeur, date_debut, date_fin)
        all_data.extend(chunk)
        current = next_month

    if not all_data:
        print(f"No data for {station_code} {grandeur}")
        return None

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates()
    os.makedirs('data/hubeau', exist_ok=True)
    output = f'data/hubeau/hubeau_{station_code}_{grandeur}.csv'
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")
    return df

def fetch_all_stations(stations_file='data/hydroportail/stations.csv'):
    stations = pd.read_csv(stations_file, sep=';')
    active = stations[stations['en_service'] == True]
    print(f"Active stations: {len(active)}")

    for _, row in active.iterrows():
        code = row['code_station']
        for grandeur in ['H', 'Q']:
            print(f"\n--- {code} {grandeur} ---")
            fetch_water_levels(code, grandeur)

if __name__ == '__main__':
    fetch_all_stations()