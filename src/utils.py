import os
import time
import pandas
import requests
from urllib.parse import urlparse, parse_qs

# ============================================================
#  UTILITAIRES
# ============================================================

def request_json_data(url, params, timeout=60): # return requested json data or raise exception
    """
    Execute HTTP GET request and return data as JSON or raise an exception.
    """
    r = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()

def request_with_pagination(url, params, pages_delay_in_seconds=0.3): # return list of all results from paginated endpoint
    """
    Parcourt toutes les pages d'un endpoint paginé Hub'Eau.
    Retourne la liste complète des résultats.
    """
    results = []
    cursor = None
    page_number = 1

    while True:  # break if cursor is None or if no more results
        pp = dict(params)
        if cursor:
            pp["cursor"] = cursor

        data_page_as_json = request_json_data(url, params=pp)
        if not data_page_as_json or "data" not in data_page_as_json:
            break

        result = data_page_as_json.get("data", [])
        results.extend(result)
        print(f"    page {page_number} → {len(result)} enregistrements (total : {len(results)})")

        # Récupérer le curseur de pagination
        next_url = data_page_as_json.get("next")
        if not next_url or not result:
            break

        qs = parse_qs(urlparse(next_url).query)
        cursor = qs.get("cursor", [None])[0]
        if not cursor:
            break

        # continue :

        page_number += 1
        time.sleep(pages_delay_in_seconds) # respecter les limites de l'API (~10 req/s)

    return results

def save_dataframe_as_csv(df, nom, type_donnee, output_dir, verbose=False) -> str:
    """
    Save pandas dataframe as csv file. Return the file's path.
    """
    if df is None or df.empty:
        return None
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{nom}_{type_donnee}.csv")
    df.to_csv(filepath, index=False, encoding="utf-8")

    filesize = os.path.getsize(filepath) / 1024
    if verbose:
        print(f">>> Sauvegardé : {filepath} ({len(df)} lignes, {filesize:.0f} Ko)")
    return filepath
