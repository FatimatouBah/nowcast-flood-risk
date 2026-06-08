from private.version import API_VERSION
from private.sites import SITES as SAMPLE_SITES
from private.utils import request_json_all

import datetime as dt

class HubeauClient:

    BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
    USER_AGENT = "jedha-dsfsft41-team3-ml-flood-forecasting-app"

    def request_stations(self) -> dict: 
        """
        Return {"api_version", "count", "stations": [{"site", "code", "label", etc.}]} 
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
        params = {"size": 100, "format": "json"}

        # TODO
        # output_fields = [ "code_site" , "code_station", "libelle_station", ... ]

        stations_codes = collect_sample_sites_stations_codes_(SAMPLE_SITES)
        params["code_station"] = ",".join(stations_codes)
        # TODO :
        # params["fields"] = ",".join(output_fields)
        json_all = request_json_all(url, params, user_agent=self.USER_AGENT)  # may throw on failure status
        assert(isinstance(json_all, dict))
        assert("data" in json_all)
        stations = [extract_station_(s) for s in json_all["data"]]
        return {
            "api_version": API_VERSION, 
            "count" : len(stations), 
            "values": stations
        }

    def request_observations(self, quantity_code: str, station_code: str, from_date: dt.date, to_date: dt.date) -> dict: 
        """
        Return {"api_version", "count", "observations": [{"ds", "y"}]} 
        """
        
        def format_date_(d_, end_of_the_day_=False):
            dt_ = dt.datetime(d_.year, d_.month, d_.day, 23, 59, 59, tzinfo=dt.timezone.utc) if end_of_the_day_ else \
                  dt.datetime(d_.year, d_.month, d_.day, 0, 0, 0, tzinfo=dt.timezone.utc)
            ds_ = dt_.isoformat().replace('+00:00', 'Z')
            return ds_
        
        def extract_observation_(o_):
            return {
                "ds": o_["date_obs_elab"],
                "y":  o_["resultat_obs_elab"],
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
        params = {"format": "json", "size": 100}

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
        json_all = request_json_all(url, params, user_agent=self.USER_AGENT)  # may throw on failure status
        assert(isinstance(json_all, dict))
        assert("data" in json_all)

        observations = [extract_observation_(o) for o in json_all["data"]]
        return {
            "api_version": API_VERSION, 
            "count" : len(observations), 
            "observations": observations
        }
