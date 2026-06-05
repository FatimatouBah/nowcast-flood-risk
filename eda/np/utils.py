import os
import time
import requests
from urllib.parse import urlparse, parse_qs

# ============================================================
#  UTILITAIRES
# ============================================================

def request_json(url, params=None, timeout_in_seconds=60): # return requested json data + elapsed time in seconds or raise exception
    """
    Execute the given API point as HTTP GET request with requested JSON output and return JSON output data or raise an exception.
    """
    headers = {"User-Agent": "jedha-dsfsft41-team3-ml-flood-forecasting-app", "Accept": "application/json"}
    response = requests.get(url, params=params, timeout=timeout_in_seconds, headers=headers)
    response.raise_for_status()
    output = response.json()  # content as json object (dictionary)
    return output, response.elapsed.total_seconds()

def request_json_all(url, params=None, requests_per_second=10, timeout_in_seconds=60, verbose=True): 
    """
    Traverse all pages of the paginated endpoint and return a distionary containing the list of collected JSON output data.
    """

    api_version = None
    count = None
    data = []

    duration_in_seconds = 0

    rate = 0
    pages = 0
    next_url = url
    complete = False
    while not complete:
        complete = True

        json_page, elapsed_in_seconds = request_json(next_url, params=params, timeout_in_seconds=timeout_in_seconds)
        if json_page:
            duration_in_seconds += elapsed_in_seconds  # elasped time is usually betwenn 250 and 750 ms.

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
            pages += 1
            rate = pages / duration_in_seconds
            if verbose:
                print(f">>> page {pages:03}: {len(page_data)} records / {page_count} at rate {rate} requests / second")

            next_url = json_page.get("next")
            if next_url:
                complete = False
                
                # respecter les limites de l'API
                if requests_per_second > 0:
                    if rate > requests_per_second:
                        time.sleep(0.3) # slow down

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
