from private.version import API_VERSION
from private.sites import SITES as SAMPLE_SITES
from private.utils import request_json_all

import datetime as dt
import pandas as pd
import numpy as np
from typing import Optional

DEFAULT_DATE_FORMAT = "%Y-%m-%d"   

WORKING_DATES_WINDOW = ("2007-01-01", "2026-12-31")
WORKING_DATES_WINDOW_AS_DATE_OBJECTS = (dt.date.fromisoformat(WORKING_DATES_WINDOW[0]), dt.date.fromisoformat(WORKING_DATES_WINDOW[1]))

PAGE_SIZE_MAX = 20000
PAGE_SIZE_DEFAULT = PAGE_SIZE_MAX // 2

QUANTITY_CODES = ("hixnj", "qixnj", "qinnj", "qmnj", "hixm", "qixm", "qinm", "qmm")

DEBUG = False

def _convert_hubeau_date_to_date_object(hubeau_datetime_as_string: Optional[str]):
    try:
        return dt.datetime.fromisoformat(hubeau_datetime_as_string).date()
    except:
        return None  

class HubeauClient:

    BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
    USER_AGENT = f"jedha-dsfsft41-team3-ml-flood-forecasting-api/{API_VERSION}"

    def request_stations(self) -> dict: 
        """
        Return {api_version, etime, count, stations: [{site, code, label, etc.}]} 
        """

        # TODO : add observable_date_first, observable_date_last

        def is_station_acceptable_(s_, working_dates_window_= WORKING_DATES_WINDOW_AS_DATE_OBJECTS):
            # L'hypothèse de travail est que chaque station de prévision a son modèle de prédiction.
            # Une station hors-service à l'instant présent ne devrait pas proposer de prédicteur dans le dashboard. 
            # En conséquence, on exclut systématiquement les stations hors services, même si elles ont pu servir à entrainer les modèles de stations en aval 
            # 
            # Si il existe une plage temporelle de travail : on rejette les stations actives qui se sont ouvertes hors de la plage de travail
            #
            acceptable_ = False

            active_ = s_["en_service"]
            if active_:
                if working_dates_window_ is None:
                    acceptable_ = True  # pas de plage de travail -> pas de limite dand la date d'ouverture -> active is enought
                else:
                    opening_date_ = s_["date_ouverture_station"]
                    opening_date_ = _convert_hubeau_date_to_date_object(opening_date_)
                    if opening_date_ is None:
                        if DEBUG:
                            print(f"...... station [{s_['code_station']}]: active with null opening date: {s_['date_ouverture_station']}")
                        acceptable_ = False
                    else:
                        upper_working_date = working_dates_window_[1]
                        assert(isinstance(upper_working_date, dt.date))
                        active_inside_dates_window_ = opening_date_ <= upper_working_date
                        if DEBUG:
                            print(f"...... station [{s_['code_station']}]: active with opening date inside working window")
                        acceptable_ = active_inside_dates_window_
            if DEBUG:
                print(f"...... station [{s_['code_station']}]: {'accepted' if acceptable_ else 'rejected'}")
            return acceptable_
        
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
        params = {
            "format": "json", 
            "size": PAGE_SIZE_DEFAULT
        }
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
        params["fields"] = ",".join(output_fields)
        response = request_json_all(url, params, user_agent=self.USER_AGENT)  # may throw on failure status
        assert(isinstance(response, dict))
        assert("etime" in response)
        etime = response["etime"]
        assert("data" in response)
        data = response["data"]
        assert(isinstance(data, list))

        # TODO use dataframe instead of list
        if DEBUG:
            print(f"****** request_stations: raw stations ({len(data)}): {data}")

        stations = [extract_station_(s) for s in data if is_station_acceptable_(s)]
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
        
        # TODO : limit the observations to the working dates window

        QUANTITY_CODE_QUALIFICATION__QUALIFIED = 20
        QUANTITY_CODE_STATUS__VALIDATED = 16  
        QUANTITY_CODE_STATUS__PREVALIDATED = 12 

        QUANTITY_CODES = ("hixnj", "qixnj", "qinnj", "qmnj", "hixm", "qixm", "qinm", "qmm")

        def format_date_(d_, end_of_the_day_=False):
            dt_ = dt.datetime(d_.year, d_.month, d_.day, 23, 59, 59, tzinfo=dt.timezone.utc) if end_of_the_day_ else \
                  dt.datetime(d_.year, d_.month, d_.day, 0, 0, 0, tzinfo=dt.timezone.utc)
            ds_ = dt_.isoformat().replace('+00:00', 'Z')
            return ds_
        
        def is_observation_acceptable_(o_, working_dates_window_= WORKING_DATES_WINDOW_AS_DATE_OBJECTS):
            qualified_ = o_["code_qualification"] == QUANTITY_CODE_QUALIFICATION__QUALIFIED
            validated_ = o_["code_statut"] in [QUANTITY_CODE_STATUS__VALIDATED, QUANTITY_CODE_STATUS__PREVALIDATED]

            observation_date_ = o_["date_obs_elab"]
            observation_date_ = _convert_hubeau_date_to_date_object(observation_date_)
            inside_window_ =  observation_date_ >= working_dates_window_[0] and observation_date_ <= working_dates_window_[1]

            return qualified_ and validated_ and inside_window_
        
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

        # TODO use dataframe instead of list

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
    
    # request quantities explicitly:

    ## daily quantitties:
    
    ### height quantitties:
    
    def request_station_hixnj_observations(self, station_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None, percentiles: Optional[list[int]] = None) -> dict: 
       return self.request_station_observations(quantity_code="hixnj", station_code=station_code, from_date=from_date, to_date=to_date, percentiles=percentiles)
        
    ### flow rate quantitties:
    
    def request_station_qixnj_observations(self, station_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None, percentiles: Optional[list[int]] = None) -> dict: 
       return self.request_station_observations(quantity_code="qixnj", station_code=station_code, from_date=from_date, to_date=to_date, percentiles=percentiles)
        
    def request_station_qinnj_observations(self, station_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None, percentiles: Optional[list[int]] = None) -> dict: 
       return self.request_station_observations(quantity_code="qinnj", station_code=station_code, from_date=from_date, to_date=to_date, percentiles=percentiles)
        
    def request_station_qmnj_observations(self, station_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None, percentiles: Optional[list[int]] = None) -> dict: 
       return self.request_station_observations(quantity_code="qmnj", station_code=station_code, from_date=from_date, to_date=to_date, percentiles=percentiles)
        
    ## monthly quantitties:
    
    ### height quantitties:
    
    def request_station_hixm_observations(self, station_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None, percentiles: Optional[list[int]] = None) -> dict: 
       return self.request_station_observations(quantity_code="hixm", station_code=station_code, from_date=from_date, to_date=to_date, percentiles=percentiles)
        
    ### flow rate quantitties:
    
    def request_station_qixm_observations(self, station_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None, percentiles: Optional[list[int]] = None) -> dict: 
       return self.request_station_observations(quantity_code="qixm", station_code=station_code, from_date=from_date, to_date=to_date, percentiles=percentiles)
        
    def request_station_qinm_observations(self, station_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None, percentiles: Optional[list[int]] = None) -> dict: 
       return self.request_station_observations(quantity_code="qinm", station_code=station_code, from_date=from_date, to_date=to_date, percentiles=percentiles)
        
    def request_station_qmm_observations(self, station_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None, percentiles: Optional[list[int]] = None) -> dict: 
       return self.request_station_observations(quantity_code="qmm", station_code=station_code, from_date=from_date, to_date=to_date, percentiles=percentiles)
        