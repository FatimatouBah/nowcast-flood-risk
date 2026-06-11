from private.hubeau_client import HubeauClient 
import datetime     as dt

if __name__=="__main__":
    hb = HubeauClient()

    show_stations = True
    show_observations = True
    
    if show_stations:
        response = hb.request_stations()
        assert(isinstance(response, dict))
        assert("stations" in response.keys())
        obs = response["stations"]
        assert("count" in response.keys())
        count = response["count"]
        assert("elapsed_seconds" in response.keys())
        elapsed_seconds = response["elapsed_seconds"]
        sample_to_display_size = min(20, len(obs))
        sample_to_display = obs[0:sample_to_display_size]
        print(f"stations: count: {count} elapsed seconds: {elapsed_seconds} obs[:{sample_to_display_size}] : {sample_to_display}")

    if show_observations:
        quantity_code = "hixnj"
        station_code = "A235020001"
        d1 = dt.date.fromisoformat("2007-01-01")
        d2 = dt.date.fromisoformat("2026-05-31")
        response = hb.request_station_observations(station_code, quantity_code="hixnj", from_date=d1, to_date=d2)
        assert(isinstance(response, dict))
        assert("observations" in response.keys())
        obs = response["observations"]
        assert("count" in response.keys())
        count = response["count"]
        assert("elapsed_seconds" in response.keys())
        elapsed_seconds = response["elapsed_seconds"]
        sample_to_display_size = min(20, len(obs))
        sample_to_display = obs[0:sample_to_display_size]
        print(f"{quantity_code} observations: at: {station_code} from: {d1} to: {d2} count: {count} elapsed seconds: {elapsed_seconds} obs[:{sample_to_display_size}] : {sample_to_display}")