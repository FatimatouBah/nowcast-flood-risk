from private.hubeau_client import HubeauClient 
import datetime     as dt

if __name__=="__main__":
    hb = HubeauClient()

    show_stations = False
    show_observations = True
    
    if show_stations:
        stations = hb.request_stations()
        print("stations: ", stations)

    if show_observations:
        quantity_code = "hixnj"
        station_code = "A235020001"
        d1 = dt.date.fromisoformat("2026-01-01")
        d2 = dt.date.fromisoformat("2026-05-31")
        obs = hb.request_observations("hixnj", station_code=station_code, starting_date=d1, ending_date=d2)
        print("observations:", quantity_code, "at:", station_code, " from:", d1, " to:", d2, ":", obs)