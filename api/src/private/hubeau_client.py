from private.version import API_VERSION
from private.sites import SITES as SAMPLE_SITES
from private.utils import request_json_all

import datetime as dt
import pandas as pd
import numpy as np
from typing import Optional

TEMPORARY_HIXNJ_TIME_PERIOD = ("2007-01-01", "2026-12-31")

PAGE_SIZE_MAX = 20000
PAGE_SIZE_DEFAULT = PAGE_SIZE_MAX // 2

DEBUG = False

class HubeauClient:

    BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
    USER_AGENT = "jedha-dsfsft41-team3-ml-flood-forecasting-app"

    def request_stations(self) -> dict: 
        """
        Return {api_version, etime, count, stations: [{site, code, label, etc.}]} 
        """

        # TODO : add observable_date_first, observable_date_last
        # TODO : reject out-of-order stations

        def is_station_active_(s_, observable_date_window_= TEMPORARY_HIXNJ_TIME_PERIOD):
            active_ = s_["en_service"]  # ignore dates for now
            return active_
        
        def extract_station_(s_):
            return {
                "site":           s_["code_site"],
                "code":           s_["code_station"],
                "label":          s_["libelle_station"],
                "river":          s_["libelle_cours_eau"],
                "region":         s_["libelle_region"],
                "department":     s_["libelle_departement"],
                "municipality":   s_["libelle_commune"],
                "longitude":      s_["longitude_station"],
                "latitude":       s_["latitude_station"],
            }
        
        def collect_sample_sites_stations_codes_(sample_sites) -> list:
            return [station["code"] for site in sample_sites.values() for station in site["stations"]]
        
        url = f"{self.BASE_URL}/referentiel/stations"
        params = {"format": "json", "size": PAGE_SIZE_DEFAULT}

        # TODO
        output_fields = [ 
            "code_site" , 
            "code_station", 
            "libelle_station", 
            "libelle_cours_eau",
            "libelle_region", 
            "libelle_departement", 
            "libelle_commune", 
            "longitude_station",
            "latitude_station",
            "en_service", 
            "date_ouverture_station",
            "date_fermeture_station"]
        stations_codes = collect_sample_sites_stations_codes_(SAMPLE_SITES)
        params["code_station"] = ",".join(stations_codes)  # all the sample's stations
        
        # TODO :
        params["fields"] = ",".join(output_fields)

        response = request_json_all(url, params, user_agent=self.USER_AGENT)  # may throw on failure status
        assert(isinstance(response, dict))
        assert("etime" in response)
        etime = response["etime"]
        assert("data" in response)
        stations = [extract_station_(s) for s in response["data"] if is_station_active_(s)]
        return {
            "api_version": API_VERSION, 
            "etime": etime, 
            "count" : len(stations), 
            "stations": stations
        }

    def request_station_dates(self, station_code: str, quantity_code: str) -> dict: 
        """
        Return {api_version, etime, dates: {lower, upper}} 
        """

        response = self.request_station_observations(station_code=station_code, quantity_code=quantity_code)
        assert(isinstance(response, dict))
        assert("etime" in response)
        assert("dates" in response)
        return {
            "api_version": API_VERSION, 
            "etime": response["etime"], 
            "dates": response["dates"]
        }

    def request_station_stats(self, station_code: str, quantity_code: str, percentiles: Optional[list[int]] = None) -> dict: 
        """
        Return {api_version, etime, dates: {lower, upper}, stats: {min, max, mean, std, q25, q50, q75, ..., q98}}
        """

        response = self.request_station_observations(station_code=station_code, quantity_code=quantity_code, percentiles=percentiles)
        assert(isinstance(response, dict))
        assert("etime" in response)
        assert("dates" in response)
        assert("stats" in response)
        return {
            "api_version": API_VERSION, 
            "etime": response["etime"], 
            "dates": response["dates"],
            "stats": response["stats"]
        }

    def request_station_observations(self, station_code: str, quantity_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None, percentiles: Optional[list[int]] = None) -> dict: 
        """
        Return {api_version, etime, count, dates: {lower, upper}, stats: {min, max, ..., q98}, observations: [{ds, yobs}]} 
        """
            
        QUANTITY_CODE_QUALIFICATION__QUALIFIED = 20
        QUANTITY_CODE_STATUS__VALIDATED = 16  
        QUANTITY_CODE_STATUS__PREVALIDATED = 12 

        DEFAULT_DATE_FORMAT = "%Y-%m-%d"   

        def format_date_(d_, end_of_the_day_=False):
            dt_ = dt.datetime(d_.year, d_.month, d_.day, 23, 59, 59, tzinfo=dt.timezone.utc) if end_of_the_day_ else \
                  dt.datetime(d_.year, d_.month, d_.day, 0, 0, 0, tzinfo=dt.timezone.utc)
            ds_ = dt_.isoformat().replace('+00:00', 'Z')
            return ds_
        
        def is_observation_acceptable_(o_):
            return o_["code_qualification"] == QUANTITY_CODE_QUALIFICATION__QUALIFIED and o_["code_statut"] in [QUANTITY_CODE_STATUS__VALIDATED, QUANTITY_CODE_STATUS__PREVALIDATED]
        
        def compute_observable_dates_(observations_):
            ds = pd.Series([dt.date.fromisoformat(o_["ds"]) for o_ in observations_])
            min, max = ds.min(), ds.max()
            datetypes = (np.datetime64, dt.date)
            is_min_a_date, is_max_a_date = isinstance(min, datetypes), isinstance(max, datetypes)
            # print(f"min: ({type(min)}) {min}")

            nat = None  # must be serializable

            min = min.strftime(DEFAULT_DATE_FORMAT) if is_min_a_date else nat
            max = max.strftime(DEFAULT_DATE_FORMAT) if is_max_a_date else nat
            return {
                "lower": min, 
                "upper": max
            }

        def compute_observable_stats_(observations_, percentiles_):
            def make_percentile_key__(nn__):
                assert(nn__ >= 0 and nn__ <= 100)
                return f"q{int(nn__):02}"
            
            # prepare stats (invalid values must be serializable (nan is not))
            stats = { "min": None, "max": None, "mean": None, "std": None }  # must be serializable (nan is not)
            local_percentiles = [25, 50, 75, 98]  #TODO must be merged with the given percentiles
            for nn in local_percentiles:
                stats[make_percentile_key__(nn)] = None

            count = len(observations_)
            if count > 0:
                yobs = pd.Series([o_["yobs"] for o_ in observations_])
                if count > 1:
                    stats["min"] = yobs.min()
                    stats["max"] = yobs.max()
                    stats["std"] = yobs.std()
                    stats["mean"] = yobs.mean()
                    for nn in local_percentiles:
                        stats[make_percentile_key__(nn)] = yobs.quantile(float(nn/100.0))
                else:            
                    for kk, _ in stats.items():         
                        stats[kk] = yobs[0]  
                    stats["std"] = 0.0
            return stats

        def extract_observation_(o_):
            return {
                "ds": o_["date_obs_elab"],
                "yobs":  o_["resultat_obs_elab"],
            }
        
        quantity_code_map = {"hixnj": "HIXnJ"}
        if quantity_code not in quantity_code_map.keys():
            raise ValueError(f"invalid quantity code: <{quantity_code}>")
        quantity_key = quantity_code_map[quantity_code]

        is_d1_defined = isinstance(from_date, dt.date) 
        is_d2_defined = isinstance(to_date, dt.date) 

        if is_d1_defined and is_d2_defined and from_date > to_date:
            raise ValueError(f"invalid dates window: first date <{from_date}> must be less or equal than last date <{to_date}>")

        ds1 = format_date_(from_date, end_of_the_day_=False) if is_d1_defined else None
        ds2 = format_date_(to_date, end_of_the_day_=True) if is_d2_defined else None

        url = f"{self.BASE_URL}/obs_elab"
        params = {"format": "json", "size": PAGE_SIZE_DEFAULT}

        output_fields = [ "date_obs_elab" , "resultat_obs_elab", "code_qualification", "code_statut" ]

        # params : 
        #   grandeur_hydro_elab
        #   code_entite
        #   date_debut_obs_elab
        #   date_fin_obs_elab
        #   fields = date_obs_elab,resultat_obs_elab,...
        params["grandeur_hydro_elab"] = quantity_key
        params["code_entite"] = station_code
        params["fields"] = ",".join(output_fields)
        if is_d1_defined:
            params["date_debut_obs_elab"] = ds1
        if is_d2_defined:
            params["date_fin_obs_elab"] = ds2
        if DEBUG:
            print(f"request_station_observations: params: {params}")
        response = request_json_all(url, params, user_agent=self.USER_AGENT)  # may throw on failure status
        assert(isinstance(response, dict))
        assert("etime" in response)
        elapseds = response["etime"]
        assert("data" in response)
        data = response["data"]
        observations = [extract_observation_(o) for o in data if is_observation_acceptable_(o)]
        dates = compute_observable_dates_(observations)
        stats = compute_observable_stats_(observations, percentiles)
        return {
            "api_version": API_VERSION, 
            "etime": elapseds, 
            "count" : len(observations), 
            "dates": dates, 
            "stats": stats, 
            "observations": observations
        }

    # -----------------------------------------------------------------------------

    # obsolete:
    # @staticmethod
    # def _compute_thresholds(observations, percentiles: list[int]) -> float:
    #     # DEBUG: print("------ inside _compute_thresholds:")
    #     # DEBUG: print(f"--------- observations: {type(observations)} ({len(observations)})")
    #     assert(isinstance(observations, dict))
    #     assert("observations" in observations.keys())
    #     # DEBUG: print(f"--------- observations: {type(observations["observations"])} ({len(observations["observations"])})")
    #     assert(isinstance(observations["observations"], list))
    #     if observations["observations"]:
    #         # DEBUG: print(f"------------ head: {type(observations["observations"][0])} ({len(observations["observations"][0])})")
    #         assert(isinstance(observations["observations"][0], dict))
    #         assert("yobs" in observations["observations"][0].keys())
    #     values = [obs["yobs"] for obs in observations["observations"]]
    #     # DEBUG: print(f"--------- values: {type(values)} ({len(values)})")
    #     thresholds = {f"q{str(percentile)}": np.percentile(values, percentile) for percentile in percentiles}
    #     return thresholds 
