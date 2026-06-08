from private.version import API_VERSION
from private.sites import SITES as SAMPLE_SITES
from private.utils import request_json_all

class HubeauClient:

    BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
    USER_AGENT = "jedha-dsfsft41-team3-ml-flood-forecasting-app"

    def request_stations(self) -> dict: 
        """
        Return {"api_version", "count", "stations": [{"code", "label", etc.}]} 
        """
        def extract_station_(s_):
            return {
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
        params = {"size": 100}

        stations_codes = collect_sample_sites_stations_codes_(SAMPLE_SITES)
        params["code_station"] = ",".join(stations_codes)
        json_all = request_json_all(url, params, user_agent=self.USER_AGENT)  # may throw on failure status
        assert(isinstance(json_all, dict))
        assert("data" in json_all)
        stations = [extract_station_(s) for s in json_all["data"]]
        return {
            "api_version": API_VERSION, 
            "count" : len(stations), 
            "stations": stations
        }
