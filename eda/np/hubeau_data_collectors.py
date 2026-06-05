"""
Hubeau's API client for hydrometric data retrieval.
"""

import pandas

from config import HUBEAU_BASE_URL
from utils import request_json_all    

# ============================================================
#  Recherche des codes de stations
# ============================================================

# TODO : tester
def request_stations(names, verbose = False): # return stations' data in a dictionary
    """
    Recherche les codes hydrométriques des stations par leur nom.
    Retourne un dict {nom: {code, libelle, cours_eau, ...}}
    """
    if verbose:
        print()
        print("="*60)
        print("Recherche des codes de stations")
        print("="*60)

    resultats = {}
    for name in names:
        print(f"\n  Recherche : {name}")
        url = f"{HUBEAU_BASE_URL}/referentiel/stations"
        params = {
            "libelle_station": name,
            "format": "json",
            "size": 10,
        }
        data = request_json_all(url, params)
        if data and data.get("data"):
            for s in data["data"]:
                print(f"    → {s['code_station']:15s} | "
                      f"{s.get('libelle_station',''):30s} | "
                      f"{s.get('libelle_cours_eau','')}")
            # Prendre la première correspondance
            s0 = data["data"][0]
            resultats[name] = {
                "code":         s0["code_station"],
                "libelle":      s0.get("libelle_station", ""),
                "cours_eau":    s0.get("libelle_cours_eau", ""),
                "departement":  s0.get("libelle_departement", ""),
                "longitude":    s0.get("longitude_station"),
                "latitude":     s0.get("latitude_station"),
            }
        else:
            print(f"    ⚠ Station non trouvée — vérifier le nom")
            resultats[name] = None

    return resultats

# ============================================================
#  Débits moyens journaliers (obs_elab — historique complet)
# ============================================================

# TODO : tester
def download_daily_flow_rates(code_station, nom, date_debut, date_fin):
    """
    Télécharge les débits moyens journaliers via /obs_elab.
    Ces données couvrent tout l'historique depuis 1900 pour certaines stations.
    Disponibles aussi bien en mode authentifié que public.
    """
    print(f"\n  Débits journaliers : {nom} ({code_station})")
    print(f"  Période : {date_debut} → {date_fin}")

    url = f"{HUBEAU_BASE_URL}/obs_elab"
    params = {
        "code_entite":          code_station,
        "date_debut_obs_elab":  date_debut,
        "date_fin_obs_elab":    date_fin,
        "grandeur_hydro_elab":  "QmJ",   # débit moyen journalier
        "format":               "json",
        "size":                 20000,
    }

    observations = request_json_all(url, params)

    if not observations:
        print(f"  ⚠ Aucune donnée retournée")
        return None

    df = pandas.DataFrame(observations)

    rename = {
        "date_obs_elab":       "date",
        "resultat_obs_elab":   "debit_ls",   # en l/s !
        "code_statut":         "statut",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "debit_ls" in df.columns:
        df["debit_m3s"] = pandas.to_numeric(df["debit_ls"], errors="coerce") / 1000

    df = df.sort_values("date")
    print(f"  ✓ {len(df)} observations récupérées")
    return df

# ============================================================
#  Observations horaires (grandeur H, temps réel)
# ============================================================

# TODO : tester
def download_hourly_water_heights(code_station, nom, date_debut, date_fin):
    """
    Télécharge les hauteurs d'eau horaires via /observations_tr.
    Les données sont brutes (statut temps réel).
    Les données "vérifiées" ne sont accessibles que via l'interface web.

    Retourne un DataFrame pandas avec colonnes : datetime, hauteur_m
    """
    print(f"\n  Hauteurs horaires : {nom} ({code_station})")
    print(f"  Période : {date_debut} → {date_fin}")

    url = f"{HUBEAU_BASE_URL}/observations_tr"
    params = {
        "code_entite":    code_station,
        "date_debut_obs": date_debut + "T00:00:00Z",
        "date_fin_obs":   date_fin   + "T23:59:59Z",
        "grandeur_hydro": "H",
        "format":         "json",
        "size":           20000,
        "timestep":       60,        # pas de temps 60 minutes
    }

    observations = request_json_all(url, params)

    if not observations:
        print(f"  ⚠ Aucune donnée retournée")
        return None

    df = pandas.DataFrame(observations)

    # Colonnes standardisées
    rename = {
        "date_obs":       "datetime",
        "resultat_obs":   "hauteur_mm",   # en mm !
        "code_statut":    "statut",
        "code_qualification": "qualification",
        "code_methode":   "methode",
    }

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "datetime" in df.columns:
        df["datetime"] = pandas.to_datetime(df["datetime"], utc=True)
    if "hauteur_mm" in df.columns:
        df["hauteur_m"] = pandas.to_numeric(df["hauteur_mm"], errors="coerce") / 1000
        df = df.drop(columns=["hauteur_mm"], errors="ignore")

    df = df.sort_values("datetime")
    print(f"  ✓ {len(df)} observations récupérées")
    return df
