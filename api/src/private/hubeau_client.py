from private.version import API_VERSION
from private.sites import SITES as SAMPLE_SITES
from private.utils import request_json_all

import datetime as dt
import numpy as np

TEMPORARY_HIXNJ_TIME_PERIOD = ("2007-01-01", "2026-06-01")

PAGE_SIZE_MAX = 20000
PAGE_SIZE_DEFAULT = PAGE_SIZE_MAX // 2

class HubeauClient:

    BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
    USER_AGENT = "jedha-dsfsft41-team3-ml-flood-forecasting-app"

    def request_stations(self) -> dict: 
        """
        Return {"api_version", "elapsed_seconds", "count", "stations": [{"site", "code", "label", etc.}]} 
        """

        def extract_station_(s_):
            return {
                "site":           s_["code_site"],
                "code":           s_["code_station"],
                "label":          s_.get("libelle_station", ""),
                "river":          s_.get("libelle_cours_eau", ""),
                "region":         s_.get("libelle_region", ""),
                "department":     s_.get("libelle_departement", ""),
                "municipality":   s_.get("libelle_commune", ""),
                "longitude":      s_.get("longitude_station"),
                "latitude":       s_.get("latitude_station"),
            }
        
        def collect_sample_sites_stations_codes_(sample_sites) -> list:
            return [station["code"] for site in sample_sites.values() for station in site["stations"]]
        
        url = f"{self.BASE_URL}/referentiel/stations"
        params = {"format": "json", "size": PAGE_SIZE_DEFAULT}

        # TODO
        # output_fields = [ "code_site" , "code_station", "libelle_station", ... ]

        stations_codes = collect_sample_sites_stations_codes_(SAMPLE_SITES)
        params["code_station"] = ",".join(stations_codes)
        
        # TODO :
        # params["fields"] = ",".join(output_fields)

        response = request_json_all(url, params, user_agent=self.USER_AGENT)  # may throw on failure status
        assert(isinstance(response, dict))
        assert("data" in response)
        stations = [extract_station_(s) for s in response["data"]]
        assert("elapsed_seconds" in response)
        elapseds = response["elapsed_seconds"]
        return {
            "api_version": API_VERSION, 
            "elapsed_seconds": elapseds, 
            "count" : len(stations), 
            "stations": stations
        }

    def request_station_dates(self, station_code: str, quantity_code: str) -> dict: 
        """
        Return {"api_version", "dates": {"lower", "upper"}} 
        """
        # TODO : 
        return {
            "api_version": API_VERSION, 
            "elapsed_seconds": 0, 
            "dates": {"lower": TEMPORARY_HIXNJ_TIME_PERIOD[0], "upper": TEMPORARY_HIXNJ_TIME_PERIOD[1]}
        }

    def request_station_thresholds(self, station_code: str, quantity_code: str, percentiles: list[int]) -> dict: 
        """
        Return {"api_version", "thresholds": {"qnn", ... (for nn in percentiles)}}
        """

        response = self.request_station_dates(station_code=station_code, quantity_code=quantity_code)    
        from_date = dt.date.fromisoformat(response["dates"]["lower"])
        to_date = dt.date.fromisoformat(response["dates"]["upper"])
        # DEBUG: print(f"****** dates: {from_date} ({type(from_date)}), {to_date} ({type(to_date)})")
        observations = self.request_station_observations(station_code=station_code, quantity_code=quantity_code, from_date=from_date, to_date=to_date)
        # DEBUG: print(f"****** observations: {type(observations)} ({len(observations)})")
        thresholds = self._compute_thresholds(observations, percentiles)
        # DEBUG: print("****** thresholds:", thresholds)
        return {
            "api_version": API_VERSION, 
            "elapsed_seconds": response["elapsed_seconds"], 
            "thresholds": thresholds
        }

    def request_station_observations(self, station_code: str, quantity_code: str, from_date: dt.date, to_date: dt.date) -> dict: 
        """
        Return {"api_version", "count", "observations": [{"ds", "yobs"}]} 
        """
        
        def format_date_(d_, end_of_the_day_=False):
            dt_ = dt.datetime(d_.year, d_.month, d_.day, 23, 59, 59, tzinfo=dt.timezone.utc) if end_of_the_day_ else \
                  dt.datetime(d_.year, d_.month, d_.day, 0, 0, 0, tzinfo=dt.timezone.utc)
            ds_ = dt_.isoformat().replace('+00:00', 'Z')
            return ds_
        
        def extract_observation_(o_):
            return {
                "ds": o_["date_obs_elab"],
                "yobs":  o_["resultat_obs_elab"],
            }
        
        quantity_code_map = {"hixnj": "HIXnJ"}
        if quantity_code not in quantity_code_map.keys():
            raise ValueError(f"invalid quantity code: <{quantity_code}>")
        quantity_key = quantity_code_map[quantity_code]

        if from_date > to_date:
            raise ValueError(f"invalid period: <{from_date}> must be less or equal to <{to_date}>")

        ds1 = format_date_(from_date, False)
        ds2 = format_date_(to_date, True)

        url = f"{self.BASE_URL}/obs_elab"
        params = {"format": "json", "size": PAGE_SIZE_DEFAULT}

        output_fields = [ "date_obs_elab" , "resultat_obs_elab" ]

        # params : 
        #   grandeur_hydro_elab
        #   code_entite
        #   date_debut_obs_elab
        #   date_fin_obs_elab
        #   fields = date_obs_elab,resultat_obs_elab,...
        params["grandeur_hydro_elab"] = quantity_key
        params["code_entite"] = station_code
        params["date_debut_obs_elab"] = ds1
        params["date_fin_obs_elab"] = ds2
        params["fields"] = ",".join(output_fields)
        response = request_json_all(url, params, user_agent=self.USER_AGENT)  # may throw on failure status
        assert(isinstance(response, dict))
        assert("data" in response)
        observations = [extract_observation_(o) for o in response["data"]]
        assert("elapsed_seconds" in response)
        elapseds = response["elapsed_seconds"]
        return {
            "api_version": API_VERSION, 
            "elapsed_seconds": elapseds, 
            "count" : len(observations), 
            "observations": observations
        }

    # -----------------------------------------------------------------------------

    @staticmethod
    def _compute_thresholds(observations, percentiles: list[int]) -> float:
        # DEBUG: print("------ inside _compute_thresholds:")
        # DEBUG: print(f"--------- observations: {type(observations)} ({len(observations)})")
        assert(isinstance(observations, dict))
        assert("observations" in observations.keys())
        # DEBUG: print(f"--------- observations: {type(observations["observations"])} ({len(observations["observations"])})")
        assert(isinstance(observations["observations"], list))
        if observations["observations"]:
            # DEBUG: print(f"------------ head: {type(observations["observations"][0])} ({len(observations["observations"][0])})")
            assert(isinstance(observations["observations"][0], dict))
            assert("yobs" in observations["observations"][0].keys())
        values = [obs["yobs"] for obs in observations["observations"]]
        # DEBUG: print(f"--------- values: {type(values)} ({len(values)})")
        thresholds = {f"q{str(percentile)}": np.percentile(values, percentile) for percentile in percentiles}
        return thresholds 
