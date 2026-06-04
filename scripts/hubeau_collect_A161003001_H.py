import requests
import pandas as pd
import os
from datetime import datetime, timedelta

STATIONS = [
    'A161003001',
    'A214010001',
    'A222000101',
    'A235020001',
    'A236003001',
    'A341020001',
    'A343021001',
    'A348020001'
]

OUTPUT_DIR = 'src/fb/hubeau'

def fetch_water_levels(station_code, grandeur='H'):
    url = 'https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr'
    all_data = []
    cursor = None

    while True:
        params = {'code_entite': station_code, 'grandeur_hydro': grandeur, 'size': 1000}
        if cursor:
            params['cursor'] = cursor
        r = requests.get(url, params=params)
        if r.status_code not in [200, 206]:
            print(f"No data — {station_code} {grandeur}")
            return None
        data = r.json()
        if not data.get('data'):
            break
        all_data.extend(data['data'])
        next_url = data.get('next')
        if not next_url or len(all_data) >= data['count']:
            break
        cursor = next_url.split('cursor=')[1].split('&')[0]

    if not all_data:
        print(f"No data — {station_code} {grandeur}")
        return None

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates()
    df['date_obs'] = pd.to_datetime(df['date_obs'])
    df = df.sort_values('date_obs', ascending=False)

    date_debut_str = df['date_obs'].min().strftime('%Y%m%d')
    date_fin_str = df['date_obs'].max().strftime('%Y%m%d')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = f'{OUTPUT_DIR}/hubeau_obstr_{station_code}_{grandeur}_{date_fin_str}_{date_debut_str}.csv'
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")
    return df

def fetch_all_stations():
    for code in STATIONS:
        for grandeur in ['H', 'Q']:
            print(f"\n--- {code} {grandeur} ---")
            fetch_water_levels(code, grandeur)

if __name__ == '__main__':
    fetch_all_stations()
