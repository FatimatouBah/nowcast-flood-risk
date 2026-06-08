from private.hubeau_client import HubeauClient 

if __name__=="__main__":
    hb = HubeauClient()
    _ = hb.request_stations()
    print(_)