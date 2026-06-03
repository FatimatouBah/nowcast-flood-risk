import os
import time
import pandas
import requests
from urllib.parse import urlparse, parse_qs

# ============================================================
#  UTILITAIRES
# ============================================================

def request_json(url, params, timeout=60): # return requested json data or raise exception
    """
    Execute the given API point as HTTP GET request with requested JSON output and return JSON output data or raise an exception.
    """
    r = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
    r.raise_for_status()
    output = r.json()  # content as json object (dictionary)
    return output

def request_json_all(url, params, next_pages_delay_in_seconds = 0.3, timeout_per_page = 60, verbose = False): 
    """
    Parcourt toutes les pages d'un endpoint paginé u.
    Retourne la liste complète des résultats.
    """

    assert(isinstance(params, dict))

    api_version = None
    count = None
    data = []
    
    cursor = None
    page_number = 1
    total_records = 0

    done = False  # break if cursor is None or if no more results
    while not done:
        done = True

        pp = dict(**params)
        if cursor:
            pp["cursor"] = cursor

        json_page = request_json(url, params=pp, timeout=timeout_per_page)
        if json_page:
            assert(isinstance(json_page, dict))
            assert(all(key in json_page for key in ("api_version", "count", "data", "next")))

            page_api_version = json_page["api_version"]
            if api_version is None:
                api_version = page_api_version
            assert(page_api_version == api_version)
            
            page_count = json_page["count"]
            if count is None:
                count = page_count
            assert(page_count == count)

            page_data = json_page.get("data", [])
            data.extend(page_data)

            # info:
            local_records = len(page_data)
            total_records += local_records
            if verbose:
                print(f">>> page {page_number} -> {local_records} enregistrements / {total_records}")

            next_url = json_page.get("next")
            if next_url:
                qs = parse_qs(urlparse(next_url).query)
                cursor = qs.get("cursor", [None])[0]
                if cursor:
                    done = False  # continue with next page

                    # info:
                    page_number += 1

                     # respecter les limites de l'API (0.3s ~> 10 req/s) 
                    if next_pages_delay_in_seconds > 0:
                        time.sleep(next_pages_delay_in_seconds)      

    return {"api_version": api_version, "count": count, "data": data}

def save_dataframe_as_csv(df, base_name, output_dirpath, verbose=False) -> str:
    """
    Save pandas dataframe as csv file. Return the file's path.
    """
    if df is None or df.empty:
        return None
    os.makedirs(output_dirpath, exist_ok=True)
    filepath = os.path.join(output_dirpath, f"{base_name}.csv")
    _ = df.to_csv(filepath, index=False, encoding="utf-8")

    filesize_in_kb = os.path.getsize(filepath) / 1024
    if verbose:
        print(f">>> Sauvegardé : {filepath} ({len(df)} lignes, {filesize_in_kb:.0f} Kb)")
    return filepath
