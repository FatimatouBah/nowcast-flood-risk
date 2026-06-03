import requests
import pandas as pd
import os

def fetch_water_levels(station_code='A161003001', grandeur='H'):
    """Fetch all water level observations from Hub'eau API"""
    
    url = 'https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr'
    all_data = []
    cursor = None

    while True:
        params = {
            'code_entite': station_code,
            'grandeur_hydro': grandeur,
            'size': 1000
        }
        if cursor:
            params['cursor'] = cursor

        r = requests.get(url, params=params)
        data = r.json()

        all_data.extend(data['data'])
        print(f"Fetched {len(all_data)} / {data['count']} observations")

        next_url = data.get('next')
        if not next_url:
            break

        cursor = next_url.split('cursor=')[1].split('&')[0]

    df = pd.DataFrame(all_data)
    os.makedirs('data/hubeau', exist_ok=True)
    df.to_csv(f'data/hubeau/{station_code}_water_levels.csv', index=False)
    print(f"Saved {len(df)} rows to data/hubeau/{station_code}_water_levels.csv")
    
    return df

if __name__ == '__main__':
    df = fetch_water_levels()