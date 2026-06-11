from private.hubeau_client import HubeauClient 
import datetime     as dt

if __name__=="__main__":
    hb = HubeauClient()

    show_stations = False
    show_statn_dates = False
    show_observations = True
    
    stations = []

    if show_stations:
        print(f"*** all stations:")
        response = hb.request_stations()
        assert(isinstance(response, dict))
        assert("etime" in response.keys())
        etime = response["etime"]
        assert("count" in response.keys())
        count = response["count"]
        assert("stations" in response.keys())
        stations = response["stations"]
        sample_to_display_size = min(20, len(stations))
        sample_to_display = stations[0:sample_to_display_size]
        print(f"stations: count: {count}, etime: {etime}, stations[:{sample_to_display_size}] : {sample_to_display}")

    if show_statn_dates:
        quantity_code = "hixnj"
        print(f"*** {quantity_code} observable dates:")
        for station in stations:
            station_code = station["code"]
            response = hb.request_station_dates(station_code, quantity_code=quantity_code)
            assert(isinstance(response, dict))
            assert("etime" in response.keys())
            etime = response["etime"]
            assert("dates" in response.keys())
            dates = response["dates"]
            print(f"station [{station_code}]: etime: {etime}, dates: {dates}")

    if show_observations:
        station_code = "A235020001" ## 
        quantity_code = "hixnj"
        d1 = dt.date.fromisoformat("2026-01-01")
        d2 = dt.date.fromisoformat("2026-12-31")
        # 1. try with defined dates:
        print(f"*** {quantity_code} observations with defined dates:")
        response = hb.request_station_observations(station_code, quantity_code="hixnj", from_date=d1, to_date=d2)
        assert(isinstance(response, dict))
        assert("etime" in response.keys())
        etime = response["etime"]
        assert("count" in response.keys())
        count = response["count"]
        assert("dates" in response.keys())
        dates = response["dates"]
        assert("stats" in response.keys())
        stats = response["stats"]
        assert("observations" in response.keys())
        obs = response["observations"]
        sample_to_display_size = min(20, len(obs))
        sample_to_display = obs[0:sample_to_display_size]
        print(f"{quantity_code} observations at {station_code} from {d1} to {d2}: etime (seconds): {etime}, count: {count}, dates: {dates}, stats: {stats}, obs[:{sample_to_display_size}] : {sample_to_display}")
        # 2. try with undefined dates:
        print(f"*** {quantity_code} observations with undefined dates:")
        response = hb.request_station_observations(station_code, quantity_code="hixnj")
        assert(isinstance(response, dict))
        assert("etime" in response.keys())
        etime = response["etime"]
        assert("count" in response.keys())
        count = response["count"]
        assert("dates" in response.keys())
        dates = response["dates"]
        assert("stats" in response.keys())
        stats = response["stats"]
        assert("observations" in response.keys())
        obs = response["observations"]
        sample_to_display_size = min(20, len(obs))
        sample_to_display = obs[0:sample_to_display_size]
        print(f"{quantity_code} observations at {station_code}: etime (seconds): {etime}, count: {count}, dates: {dates}, stats: {stats}, obs[:{sample_to_display_size}] : {sample_to_display}")