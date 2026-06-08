# -*- coding=utf-8 -*-
# CONFIG.utils.py
import json
import pandas as pd
from pathlib import Path
import requests
# Import regex locally so only this function is changed and self-contained.
import re
# Import datetime locally to validate date values after regex extraction.
from datetime import datetime
# Import 
from io import StringIO, BytesIO


# *==========================================================================================
# * Fonctions utiles d'AWS S3 pour la gestion et la sauvegarde des données
# *==========================================================================================

def get_extension_from_filepath(file_path):
    """
    Get the file extension from a file path or file name.
    
    Parameters:
    - file_path: str, the file path or file name from which to extract the extension
    
    Returns:
    - str, the file extension (without the dot) or None if no extension is found
    """
    if isinstance(file_path, (str, Path)):
        # Use Path to get the suffix (extension) of the file path
        extension = Path(file_path).suffix
        if extension:
            return extension.lstrip(".") # remove the dot from the extension
        else:
            return None
    else:
        return None

def upload_file_to_s3_bucket(data_file_path: str | object, s3_bucket_resource: object, s3_storage_folder: str, Key_file_basename: str = "csv") -> str | None:
    """
    Upload a file to an S3 bucket using the upload_file() method of the S3 Bucket resource.
    
    Parameters:
    - data_file_path: str, the local path of the file to be uploaded
    - s3_bucket_resource: boto3 S3 Bucket resource object, the S3 bucket where to upload the file
    - s3_storage_folder: str, the folder in the S3 bucket where to upload the file (should include the root folder if it exists)
    - Key_file_basename: str, the format of the file to be uploaded (default is "csv") if the data_file_path is not a file path but directly the data content to be stored in s3, then the format should be specified (csv or json) to use the appropriate method for uploading the data content to s3.
    
    Returns:
    - str, the URI of the uploaded file in the S3 bucket or None if the upload fails
    
    Notes: data_file_path and Key_file_basename can't be used together, if data_file_path is a file path, the Key_file_basename will be determined automatically from the file extension, if data_file_path is directly the data content to be stored in s3, then the Key_file_basename should be specified (csv or json) to use the appropriate method for uploading the data content to s3.
    """
    # Wrap the upload call in a try-except block to gracefully handle network timeouts or permission errors
    try:
        # data_file_path is a file path or any object file path
        if isinstance(data_file_path, (str, Path)) and Path(data_file_path).is_file():
            # Use Path to get the base name of the file path to use it as object key when uploading to s3
            file_basename = Path(data_file_path).name # = os.path.basename(data_file_path)
            # Perform the actual file upload operation via boto3, transferring the file to S3
            # Upload the file to s3 using upload_file() method of the S3 Bucket resource
            s3_key_file = f"{s3_storage_folder}/{file_basename}"
            # upload the file to s3 using the upload_file() method of the S3 Bucket resource, which takes the local file path and the S3 key (including the folder path in the bucket) as parameters
            s3_bucket_resource.upload_file(Filename=str(data_file_path), Key=s3_key_file)
        else: # data_file_path is directly the data content to be stored in s3, then the Key_file_basename should be specified (csv or json) to use the appropriate method for uploading the data content to s3.
            data = data_file_path
            key_name = str(Key_file_basename)
            if key_name in ("csv", "json"):
                key_name = f"data.{key_name}"
            s3_key_file = f"{s3_storage_folder}/{key_name}"
            # directly store an object such as dataframe or other data structures in s3 using put_object() method of the S3 Object resource
            if key_name.endswith(".csv"):
                if isinstance(data, pd.DataFrame):
                    body = data.to_csv(index=False)
                elif isinstance(data, (dict, list)):
                    body = pd.DataFrame(data).to_csv(index=False)
                else:
                    body = str(data)
                # buffer = StringIO()
                # data.to_csv(buffer, index=False)
                # body = buffer.getvalue()
            elif key_name.endswith(".json") and isinstance(data, pd.DataFrame):
                # body = data.to_json(orient="records", indent=4).encode("utf-8")
                body = data.to_json(orient="records", indent=4)
            elif key_name.endswith(".json"):
                # body=json.dumps(data).encode("utf-8")
                body = json.dumps(data, indent=4, default=str)
            elif Key_file_basename == "bytes" or isinstance(data, (bytes, bytearray)):
                body = data  # already bytes
            else:
                body = str(data)
            # upload the data content to s3 using the put_object() method of the S3 Object resource, which takes the data content as Body and the S3 key (including the folder path in the bucket) as parameters
            s3_bucket_resource.put_object(Body=body, Key=s3_key_file)
        
        #check if the file is uploaded successfully by listing the objects in the bucket and checking if the uploaded file is there
        # print(f"\nObjects in the bucket {s3_bucket_resource.name} after uploading {file_basename}:")
        for obj in s3_bucket_resource.objects.all():
            if obj.key.startswith(s3_key_file):
                s3_URI = f"s3://{s3_bucket_resource.name}/{obj.key}"
                # print(f" - object - Key: {obj.key}, Size: {obj.size} bytes, Last Modified: {obj.last_modified}")
                # Print a success confirmation upon completion
                print("  Success!")
                # print(f"   URI: {s3_URI}")
                return s3_URI
    except Exception as error_msg:
        # Catch the exception and print the exact error message if the upload fails
        print(f"  Error: {error_msg}")
        # return URI of the uploaded file in s3
        return None


# ==========================================================================================
# Fonctions utiles pour ETL des données d'observations hydrologiques
# ==========================================================================================

def save_json_file(data, json_output_filepath):
    with open(json_output_filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)


def _fetch_api_json(obs_url, params, timeout_seconds=45):
    """Call Hub'Eau API and always return a dict to keep ETL flow resilient."""
    # Try to execute the HTTP request to the API endpoint.
    try:
        # Send GET request with query parameters and a timeout to avoid hanging forever.
        response = requests.get(obs_url, params=params, timeout=timeout_seconds)
    # Catch any network, timeout, DNS, SSL, or connection-related requests exception.
    except requests.RequestException as exc:
        # Log the request failure for troubleshooting without interrupting the ETL loop.
        print(f"  API request error: {exc}")
        # Return a safe default payload with no data so caller can continue processing.
        return {
            # Explicitly state no rows are available.
            "count": 0,
            # Keep data shape consistent with expected successful responses.
            "data": [],
            # Preserve error context for diagnostics.
            "error": f"request_error: {exc}",
        }

    # If HTTP status is not in 2xx range, log a short diagnostic message.
    if not response.ok:
        # Build a compact preview of response body (first 300 chars, single-line).
        preview = response.text[:300].replace("\n", " ").replace("\r", " ")
        # Print status, content type, and text preview to understand API-side failures.
        print(
            "  API non-200 response "
            f"(status={response.status_code}, content_type={response.headers.get('Content-Type', '')}): {preview}"
        )

    # Try to decode the HTTP response body as JSON.
    try:
        # Parse JSON payload returned by API.
        data = response.json()
    # Catch JSON parsing errors when response is empty, HTML, or malformed JSON.
    except ValueError:
        # Build a short sanitized preview of raw text response for debugging.
        preview = response.text[:300].replace("\n", " ").replace("\r", " ")
        # Log explicit decode failure details.
        print(
            "  API returned a non-JSON/empty response "
            f"(status={response.status_code}, content_type={response.headers.get('Content-Type', '')}): {preview}"
        )
        # Return fallback structure instead of raising, to keep ETL robust.
        return {
            # No valid JSON records available.
            "count": 0,
            # Keep data key present with empty list.
            "data": [],
            # Tag error type for downstream logs/analysis.
            "error": "invalid_json_response",
            # Preserve HTTP status for diagnostics.
            "status_code": response.status_code,
            # Preserve content type returned by server.
            "content_type": response.headers.get("Content-Type", ""),
            # Preserve response snippet for quick triage.
            "response_preview": preview,
        }

    # Ensure decoded JSON is an object/dict because callers expect dict-like access.
    if not isinstance(data, dict):
        # Log unexpected JSON top-level type to help identify API contract changes.
        print(f"  API JSON payload is not a dict (type={type(data).__name__}); skipping record.")
        # Return safe empty structure so processing can continue.
        return {
            # Signal no usable rows.
            "count": 0,
            # Provide empty data array.
            "data": [],
            # Keep reason of rejection for debugging.
            "error": f"unexpected_payload_type:{type(data).__name__}",
            # Include status to correlate with transport-level behavior.
            "status_code": response.status_code,
        }

    # Some valid payloads may omit count; normalize it for downstream consistency.
    if "count" not in data:
        # Compute count from data length when data is a list, else default to zero.
        data["count"] = len(data.get("data", [])) if isinstance(data.get("data"), list) else 0
    # Return normalized dictionary payload to caller.
    return data
        
# ****************************
# *FONCTION RECUP ALL OBSERVATIONS historiques: all quantities (QmnJ, QmM, HIXM, HIXnJ, QINM, QINnJ, QixM, QIXnJ) in one csv file for each station 
# ****************************


def api_get_obs_elab_ALL_data(API_BASE_URL, code_site, code_station, site_name="", date_debut_obs="2007-01-01", date_fin_obs="2026-06-02", local_data_dir=None,
                              save_in_aws_s3=True, s3_bucket_resource=None, s3_storage_vars=None):
    """
    Fonction pour récupérer observations hydrométriques
    pour une station donnée
    
    End Point: /api/v2/hydrometrie/obs_elab
    
    - Grandeurs hydrométriques (grandeur_hydro_elab) élaborées disponibles : 
        - débits moyens journaliers (QmnJ), 
        - débits moyens mensuels (QmM), 
        - Hauteur instantanée maximale mensuelle (HIXM), 
        - Hauteur instantanée maximale journalière (HIXnJ), 
        - Débit instantané minimal mensuel (QINM), 
        - Débit instantané minimal journalier (QINnJ), 
        - Débit instantané maximal mensuel (QixM), 
        - Débit instantané maximal journalier (QIXnJ)
    
    Notes: Si la valeur du paramètre size n'est pas renseignée, la taille de page par défaut : 1000, taille max de la page : 20000.
    -> La profondeur d'accès aux résultats est : 20000, calcul de la profondeur = numéro de la page * nombre maximum de résultats dans une page.
    -> Trie par défaut : code_station, date_obs_elab asc

Parametres:
    
    - code_entite : array[string] (query) : Code Sandre des sites ou stations hydrométriques, si plusieurs codes les séparer par une virgule, nombre maximum de codes = 100. 
                    Possibilité d'utiliser un pattern (ex: K*)

    - date_debut_obs_elab   : string($date-time) (query) :: (default = 20) : Date de début des observations élaborées
    - date_debut_prod       : string($date-time) (query) :: (default = 20) : Date de début de plage d'intégration des données dans Hub'eau
    - date_fin_prod         : string($date-time) (query) :: (default = 20) : Date de fin de plage d'intégration des données dans Hub'eau
    - date_fin_obs_elab     : string($date-time) (query) :: (default = 20) : Date de fin des observations élaborées
    - distance              : number($double) (query) :: (default = 30) : Rayon de recherche en kilomètre, le point doit être utilisé comme séparateur décimal, exemple : 30
    - fields                : string (query) :: (default = 20) : Liste des champs souhaités dans la réponse (fonctionnalité expérimentale), par exemple fields=code_station,localisation
    - format                : string (query) :: (default = json) : Format de réponse attendu. Supportés : json, geojson (défaut : json)
    - grandeur_hydro_elab : string (query) :: (default = *) : Type de grandeur hydrométrique élaborée (HIXM, HIXnJ, QINM, QINnJ, QixM, QIXnJ, QmM ou QmnJ)
    - latitude : number($double) (query) :: (default = 20) : Latitude du point en WGS84 pour la recherche par rayon, le point doit être utilisé comme séparateur décimal, exemple : 47.829
    - longitude : number($double) (query) :: (default = 20) : -Longitude du point en WGS84 pour la recherche par rayon, le point doit être utilisé comme séparateur décimal, exemple : 1.937
    - resultat_max : number($double) (query) :: (default = 20) : Valeur maximale du résultat : renvoie tous les résultats dont resultat_obs_elab <= resultat_max
    - resultat_min : number($double) (query) :: (default = 20) : Valeur minimale du résultat : renvoie tous les résultats dont resultat_obs_elab >= resultat_min
    - size : integer($int32) (query) :: (default = 20) : Nombre maximum de résultats dans une page  
    Returns:
    - df: pandas DataFrame contenant les observations élaborées récupérées pour la station cible et la période donnée
    - csv_output_filename: str, le nom du fichier csv de sortie contenant les observations élaborées nettoyées
    - json_output_filename: str, le nom du fichier json de sortie contenant les observations élaborées nettoyées
    - s3_csv_uri: str, l'URI du fichier csv nettoyé uploadé dans le bucket s3 (None si l'upload échoue ou si save_in_aws_s3=False)
    - s3_json_uri: str, l'URI du fichier json nettoyé uploadé dans le bucket s3 (None si l'upload échoue ou si save_in_aws_s3=False)
    """
    
    # créer dossier data/all qui contient TOUTES les grandeurs hydrométriques élaborées pour chaque station
    output_cleaned_dir, output_raw_dir, output_dir = Path(""), Path(""), Path("")
    if local_data_dir not in (None, "", False, 0): 
        if Path(f"{local_data_dir}").exists() == False:
            Path(f"{local_data_dir}").mkdir(parents=True, exist_ok=True)
            # os.mkdir(f"{local_data_dir}")
        # output_dir = os.path.join(f"{local_data_dir}", "all")
        output_dir = Path(f"{local_data_dir}") / "all"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_raw_dir = output_dir / "raw"
        output_cleaned_dir = output_dir / "cleaned"
        output_raw_dir.mkdir(parents=True, exist_ok=True)
        output_cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # endpoint observations
    obs_url = f"{API_BASE_URL}/obs_elab"

    # paramètres de requête
    if site_name in ("Colmar-1",):
        params = {
            "code_entite": code_station,   # station cible
            # "date_debut_obs_elab": date_debut_obs,  # date début
            "date_fin_obs_elab": date_fin_obs,      # date fin
            "size": 20000,                 # taille max page
            "format": "json",               # format réponse en json (par defaut)
        }
    else:
        params = {
            "code_entite": code_station,   # station cible
            # "date_debut_obs_elab": date_debut_obs,  # date début
            "date_fin_obs_elab": date_fin_obs,      # date fin
            "size": 20000,                 # taille max page
            "format": "json",               # format réponse en json (par defaut)
        }
    
    # appel API + parsing JSON robuste
    data = _fetch_api_json(obs_url, params=params)

    # Save the raw JSON response to a file for debugging
    if site_name in ("", None):
        path_format = f"hubeau_obs_elab_{code_site}_{code_station}_ALL_{date_debut_obs}_{date_fin_obs}"
        raw_json_output_filename = f"{path_format}.json"
    else:
        path_format = f"hubeau_obs_elab_{site_name}_{code_site}_{code_station}_ALL_{date_debut_obs}_{date_fin_obs}"
        raw_json_output_filename = f"{path_format}.json"
    if local_data_dir not in (None, "", False, 0):
        save_json_file(data, output_raw_dir / raw_json_output_filename)
        if save_in_aws_s3: #json.dumps(data_file_path)
            upload_file_to_s3_bucket(str(output_raw_dir / raw_json_output_filename), s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/raw")
    else: # if no local data dir specified, directly upload the data content to s3 without saving it locally first, in this case the Key_file_basename should be specified (csv or json) to use the appropriate method for uploading the data content to s3.
        upload_file_to_s3_bucket(data, s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/raw", Key_file_basename=f"{path_format}.json")
    
    if data.get("count", 0) != 0:
        # vérifier si données présentes et les convertir en dataframe
        if "data" in data:
            df = pd.DataFrame(data["data"])
            start_date = df["date_obs_elab"].min() # get the minimum date of the observations
            end_date = df["date_obs_elab"].max() # get the maximum date of the observations
            # start_date = start_date.replace("-", "").replace(" ", "") # formatage date pour nom fichier
            # end_date = end_date.replace("-", "").replace(" ", "") # formatage
            start_date = start_date.replace(" ", "") # formatage date pour nom fichier
            end_date = end_date.replace(" ", "") # formatage
            # path format = f"hubeau_obs_elab_{site_name}_{code_site}_{code_station}_ALL_{start_date}_{end_date}"
            if site_name in ("", None):
                path_format = f"hubeau_obs_elab_{code_site}_{code_station}_ALL_{start_date}_{end_date}"
                csv_output_filename = f"{path_format}.csv"
                json_output_filename = f"{path_format}.json"
            else:
                path_format = f"hubeau_obs_elab_{site_name}_{code_site}_{code_station}_ALL_{start_date}_{end_date}"
                csv_output_filename = f"{path_format}.csv"
                json_output_filename = f"{path_format}.json"
            s3_csv_uri, s3_json_uri = None, None
            if local_data_dir not in (None, "", False, 0): 
                df.to_csv(output_cleaned_dir / csv_output_filename, index=False)
                df.to_json(output_cleaned_dir / json_output_filename, orient="records", indent=4)
                if save_in_aws_s3:
                    s3_csv_uri = upload_file_to_s3_bucket(str(output_cleaned_dir / csv_output_filename), s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/cleaned")
                    s3_json_uri = upload_file_to_s3_bucket(str(output_cleaned_dir / json_output_filename), s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/cleaned")
            else:
                s3_csv_uri = upload_file_to_s3_bucket(df, s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/cleaned", Key_file_basename=f"{path_format}.csv")
                s3_json_uri = upload_file_to_s3_bucket(df, s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/cleaned", Key_file_basename=f"{path_format}.json")
            return df, csv_output_filename, json_output_filename, s3_csv_uri, s3_json_uri 
        else:
            return pd.DataFrame(), None, None, None, None
    else:
        return pd.DataFrame(), None, None, None, None


# ****************************
# *FONCTION RECUP OBSERVATIONS historiques for each Qand QmM, HIXM, HIXnJ, QINM, QINnJ, QixM, QIXnJ in each station
# ****************************

def api_get_obs_elab_data(API_BASE_URL, code_site, code_station, site_name="", date_debut_obs="2007-01-01", date_fin_obs="2026-06-02", grandeur_hydro_elab="HIXM", local_data_dir=None,
                          save_in_aws_s3=True, s3_bucket_resource=None, s3_storage_vars=None):
    """
    Fonction pour récupérer observations hydrométriques
    pour une station donnée
    
    End Point: /api/v2/hydrometrie/obs_elab
    
    - Grandeurs hydrométriques (grandeur_hydro_elab) élaborées disponibles : 
        - débits moyens journaliers (QmnJ), 
        - débits moyens mensuels (QmM), 
        - Hauteur instantanée maximale mensuelle (HIXM), 
        - Hauteur instantanée maximale journalière (HIXnJ), 
        - Débit instantané minimal mensuel (QINM), 
        - Débit instantané minimal journalier (QINnJ), 
        - Débit instantané maximal mensuel (QixM), 
        - Débit instantané maximal journalier (QIXnJ)
    
    Notes: Si la valeur du paramètre size n'est pas renseignée, la taille de page par défaut : 1000, taille max de la page : 20000.
    -> La profondeur d'accès aux résultats est : 20000, calcul de la profondeur = numéro de la page * nombre maximum de résultats dans une page.
    -> Trie par défaut : code_station, date_obs_elab asc

Parametres:
    
    - code_entite : array[string] (query) : Code Sandre des sites ou stations hydrométriques, si plusieurs codes les séparer par une virgule, nombre maximum de codes = 100. 
                    Possibilité d'utiliser un pattern (ex: K*)

    - date_debut_obs_elab   : string($date-time) (query) :: (default = 20) : Date de début des observations élaborées
    - date_debut_prod       : string($date-time) (query) :: (default = 20) : Date de début de plage d'intégration des données dans Hub'eau
    - date_fin_prod         : string($date-time) (query) :: (default = 20) : Date de fin de plage d'intégration des données dans Hub'eau
    - date_fin_obs_elab     : string($date-time) (query) :: (default = 20) : Date de fin des observations élaborées
    - distance              : number($double) (query) :: (default = 30) : Rayon de recherche en kilomètre, le point doit être utilisé comme séparateur décimal, exemple : 30
    - fields                : string (query) :: (default = 20) : Liste des champs souhaités dans la réponse (fonctionnalité expérimentale), par exemple fields=code_station,localisation
    - format                : string (query) :: (default = json) : Format de réponse attendu. Supportés : json, geojson (défaut : json)
    - grandeur_hydro_elab : string (query) :: (default = *) : Type de grandeur hydrométrique élaborée (HIXM, HIXnJ, QINM, QINnJ, QixM, QIXnJ, QmM ou QmnJ)
    - latitude : number($double) (query) :: (default = 20) : Latitude du point en WGS84 pour la recherche par rayon, le point doit être utilisé comme séparateur décimal, exemple : 47.829
    - longitude : number($double) (query) :: (default = 20) : -Longitude du point en WGS84 pour la recherche par rayon, le point doit être utilisé comme séparateur décimal, exemple : 1.937
    - resultat_max : number($double) (query) :: (default = 20) : Valeur maximale du résultat : renvoie tous les résultats dont resultat_obs_elab <= resultat_max
    - resultat_min : number($double) (query) :: (default = 20) : Valeur minimale du résultat : renvoie tous les résultats dont resultat_obs_elab >= resultat_min
    - size : integer($int32) (query) :: (default = 20) : Nombre maximum de résultats dans une page  
    Returns:
    - df: pandas DataFrame contenant les observations élaborées récupérées pour la station cible et la période donnée
    - csv_output_filename: str, le nom du fichier csv de sortie contenant les observations élaborées nettoyées
    - json_output_filename: str, le nom du fichier json de sortie contenant les observations élaborées nettoyées
    - s3_csv_uri: str, l'URI du fichier csv nettoyé uploadé dans le bucket s3 (None si l'upload échoue ou si save_in_aws_s3=False)
    - s3_json_uri: str, l'URI du fichier json nettoyé uploadé dans le bucket s3 (None si l'upload échoue ou si save_in_aws_s3=False)
    """
    output_cleaned_dir, output_raw_dir, output_dir = Path(""), Path(""), Path("")
    # créer dossier data/vars qui contient chaque grandeur hydrométrique élaborée pour chaque station
    if local_data_dir not in (None, ""):
        if Path(f"{local_data_dir}").exists() == False:
            Path(f"{local_data_dir}").mkdir(parents=True, exist_ok=True)
        output_dir = Path(f"{local_data_dir}") / "vars"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_raw_dir = output_dir / "raw"
        output_cleaned_dir = output_dir / "cleaned"
        output_raw_dir.mkdir(parents=True, exist_ok=True)
        output_cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # endpoint observations
    obs_url = f"{API_BASE_URL}/obs_elab"

    # paramètres de requête
    if site_name in ("Colmar-1",):
        params = {
            "code_entite": code_station,   # station cible
            # "date_debut_obs_elab": date_debut_obs,  # date début
            # "date_fin_obs_elab": date_fin_obs,      # date fin
            "size": 20000,                 # taille max page
            "format": "json",               # format réponse
            "grandeur_hydro_elab": grandeur_hydro_elab,  # grandeur hydrométrique élaborée
        }
    else:
        params = {
            "code_entite": code_station,   # station cible
            "date_debut_obs_elab": date_debut_obs,  # date début
            # "date_fin_obs_elab": date_fin_obs,      # date fin
            "size": 20000,                 # taille max page
            "format": "json",               # format réponse
            "grandeur_hydro_elab": grandeur_hydro_elab,  # grandeur hydrométrique élaborée
        }
    
    # appel API + parsing JSON robuste
    data = _fetch_api_json(obs_url, params=params)
    # pprint(data)
    # Save the raw JSON response to a file for debugging
    if site_name in ("", None):
        path_format = f"hubeau_obs_elab_{code_site}_{code_station}_{grandeur_hydro_elab}_{date_debut_obs}_{date_fin_obs}"
        raw_json_output_filename = f"{path_format}.json"
    else:
        path_format = f"hubeau_obs_elab_{site_name}_{code_site}_{code_station}_{grandeur_hydro_elab}_{date_debut_obs}_{date_fin_obs}"
        raw_json_output_filename = f"{path_format}.json"
    if local_data_dir not in (None, ""): 
        save_json_file(data, output_raw_dir / raw_json_output_filename)
        if save_in_aws_s3:
            upload_file_to_s3_bucket(str(output_raw_dir / raw_json_output_filename), s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/raw")  
    else: # if no local data dir specified, directly upload the data content to s3 without saving it locally first, in this case the Key_file_basename should be specified (csv or json) to use the appropriate method for uploading the data content to s3.    
        upload_file_to_s3_bucket(data, s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/raw", Key_file_basename=f"{path_format}.json")  
    
    if data.get("count", 0) != 0:
        # vérifier si données présentes et les convertir en dataframe
        if "data" in data:
            df = pd.DataFrame(data["data"])
            start_date = df["date_obs_elab"].min() # get the minimum date of the observations
            end_date = df["date_obs_elab"].max() # get the maximum date of the observations
            # print(f"==> Observations élaborées récupérées pour station {code_station} \t- Grandeur:{grandeur_hydro_elab} :: {len(df)} lignes, de {start_date} à {end_date}")
            # start_date = start_date.replace("-", "").replace(" ", "") # formatage date pour nom fichier
            # end_date = end_date.replace("-", "").replace(" ", "") # formatage
            start_date = start_date.replace(" ", "") # formatage date pour nom fichier
            end_date = end_date.replace(" ", "") # formatage
            # path_format = f"hubeau_obs_elab_{site_name}_{code_site}_{code_station}_HIXnJ_{start_date}_{end_date}"
            if site_name in ("", None):
                path_format = f"hubeau_obs_elab_{code_site}_{code_station}_{grandeur_hydro_elab}_{start_date}_{end_date}"
                csv_output_filename = f"{path_format}.csv"
                json_output_filename = f"{path_format}.json"
            else:
                path_format = f"hubeau_obs_elab_{site_name}_{code_site}_{code_station}_{grandeur_hydro_elab}_{start_date}_{end_date}"
                csv_output_filename = f"{path_format}.csv"
                json_output_filename = f"{path_format}.json"
            
            s3_csv_uri, s3_json_uri = None, None
            if local_data_dir not in (None, ""): 
                df.to_csv(output_cleaned_dir / csv_output_filename, index=False)
                df.to_json(output_cleaned_dir / json_output_filename, orient="records", indent=4)
                if save_in_aws_s3:
                    s3_csv_uri = upload_file_to_s3_bucket(str(output_cleaned_dir / csv_output_filename), s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/cleaned")
                    s3_json_uri = upload_file_to_s3_bucket(str(output_cleaned_dir / json_output_filename), s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/cleaned")
            else:
                s3_csv_uri = upload_file_to_s3_bucket(df, s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/cleaned", Key_file_basename=f"{path_format}.csv")
                s3_json_uri = upload_file_to_s3_bucket(df, s3_bucket_resource, s3_storage_folder=f"{s3_storage_vars}/cleaned", Key_file_basename=f"{path_format}.json")
            return df, csv_output_filename, json_output_filename, s3_csv_uri, s3_json_uri
        else:
            return pd.DataFrame(), None, None, None, None
    else:
        return pd.DataFrame(), None, None, None, None


# *==========================================================================================
# *Fonctions to get or download URI data from S3 and load it into a pandas dataframe
# *==========================================================================================

def convert_date_to_datetime(date_str):
    """Convert a date string in the format YYYYMMDD to a datetime string in the format YYYY-MM-DD.
    Args:
        date_str (str): Date string in the format YYYYMMDD.

    Returns:
        str: Date string in the format YYYY-MM-DD.
    """
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"


# Extract metadata (site name, codes, measure, dates) from S3 URI filename
# Supports both date formats: "YYYY-MM-DD" and "YYYYMMDD" (with .csv or .json extension)
def parse_s3_uri(s3_uri):
    """
    Parse an S3 URI filename and extract metadata fields.

    Supported filename patterns:
    - raw_hubeau_obs_elab_<site_name>_<code_site>_<code_station>_<measure>_<start_date>_<end_date>.<csv|json>
    - hubeau_obs_elab_<site_name>_<code_site>_<code_station>_<measure>_<start_date>_<end_date>.<csv|json>

    Notes:
    - <site_name> can contain underscores.
    - Dates can be in YYYY-MM-DD or YYYYMMDD format.
    - Returned dates are normalized to YYYY-MM-DD.

    Args:
        s3_uri (str): Full S3 object URI.

    Returns:
        tuple[str, str, str, str, str, str]:
            (site_name, code_site, code_station, measure, start_date, end_date)
    """

    # # Import regex locally so only this function depends on it.
    # import re

    # # Import datetime locally to validate dates after regex extraction.
    # from datetime import datetime

    # Define one regex that supports both prefixes, both date formats, and both extensions.
    pattern = re.compile(
        r"^(?:raw_)?hubeau_obs_elab_"
        r"(?P<site_name>.+?)_"
        r"(?P<code_site>[A-Za-z0-9-]+)_"
        r"(?P<code_station>[A-Za-z0-9-]+)_"
        r"(?P<measure>[A-Za-z0-9-]+)_"
        r"(?P<start_date>\d{4}-\d{2}-\d{2}|\d{8})_"
        r"(?P<end_date>\d{4}-\d{2}-\d{2}|\d{8})"
        r"\.(?:csv|json)$"
    )

    # Create a small helper that validates a date string with strict calendar checking.
    def _validate_date(date_value):
        # Parse with dash format when the date contains '-' separators.
        if "-" in date_value:
            # Validate true calendar date (month/day correctness included).
            datetime.strptime(date_value, "%Y-%m-%d")
        # Otherwise parse compact format with 8 digits.
        else:
            # Validate compact calendar date (month/day correctness included).
            datetime.strptime(date_value, "%Y%m%d")

    try:
        # Validate the input type early to avoid string operation errors.
        if not isinstance(s3_uri, str):
            # Raise a clear message if caller passes a non-string value.
            raise ValueError(f"S3 URI must be a string, got {type(s3_uri).__name__}")

        # Remove accidental spaces around the URI before processing.
        s3_uri = s3_uri.strip()

        # Reject empty input after trimming spaces.
        if not s3_uri:
            # Raise a direct message for empty URI values.
            raise ValueError("S3 URI is empty")

        # Extract only the filename from the full S3 path.
        file_name = s3_uri.rsplit("/", 1)[-1]

        # Match the filename against the supported naming convention.
        match = pattern.match(file_name)

        # Stop with an explicit message if the filename does not match expected schema.
        if not match:
            # Explain the expected formats in a concise and actionable way.
            raise ValueError(
                "Invalid filename format. Expected 'raw_hubeau_obs_elab_...' or 'hubeau_obs_elab_...' with .csv/.json"
            )

        # Extract all captured groups as a dictionary for readable access.
        fields = match.groupdict()

        # Validate start_date to ensure it is a real date, not just a regex-shaped string.
        _validate_date(fields["start_date"])

        # Validate end_date to ensure it is a real date, not just a regex-shaped string.
        _validate_date(fields["end_date"])

        # Convert compact start_date (YYYYMMDD) to normalized format (YYYY-MM-DD).
        start_date = convert_date_to_datetime(fields["start_date"]) if len(fields["start_date"]) == 8 else fields["start_date"]

        # Convert compact end_date (YYYYMMDD) to normalized format (YYYY-MM-DD).
        end_date = convert_date_to_datetime(fields["end_date"]) if len(fields["end_date"]) == 8 else fields["end_date"]

        # Return the values in the exact tuple order used by the rest of the script.
        return (
            fields["site_name"],
            fields["code_site"],
            fields["code_station"],
            fields["measure"],
            start_date,
            end_date,
        )

    except (TypeError, ValueError) as e:
        # Add URI context to every parsing error so debugging remains straightforward.
        raise ValueError(f"Error parsing S3 URI '{s3_uri}': {str(e)}")

if __name__ == "__main__":
    pass