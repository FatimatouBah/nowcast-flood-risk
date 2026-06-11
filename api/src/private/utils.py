import requests
import time

DEBUG = False

def request_json(url, params=None, user_agent=None, timeout_in_seconds=60): # return requested json data + elapsed time in seconds or raise exception
    """
    Execute the given API point as HTTP GET request with requested JSON output and return JSON output data or raise an exception.
    """

    headers = {"Accept": "application/json"}
    if user_agent:
        headers["User-Agent"] = user_agent  # "jedha-dsfsft41-team3-ml-flood-forecasting-app"

    response = requests.get(url, params=params, timeout=timeout_in_seconds, headers=headers)
    response.raise_for_status()
    output = response.json()  # content as json object (dictionary)
    return output, response.elapsed.total_seconds()

DELAY_BETWEEN_PAGES = 1.0  # seconds

def request_json_all(url, params=None, user_agent=None, requests_per_second=10, timeout_in_seconds=60):  # return {"api_version", "count", "data": [json-data]} or raise exception
    """
    Traverse all pages of the paginated endpoint and return a distionary containing the list of collected JSON output data.
    """

    api_version = None
    count = None
    data = []

    duration_in_seconds = 0

    rate = 0
    pages = 0
    next_url = url
    next_params = params
    complete = False
    while not complete:
        complete = True

        json_page, elapsed_in_seconds = request_json(next_url, params=next_params, user_agent=user_agent, timeout_in_seconds=timeout_in_seconds)
        if json_page:
            duration_in_seconds += elapsed_in_seconds  # elasped time is usually betwenn 250 and 750 ms.

            assert(isinstance(json_page, dict))
            assert(all(key in json_page for key in ("api_version", "count", "data", "next")))

            page_api_version = json_page["api_version"]
            if api_version is None:
                api_version = page_api_version
            assert(page_api_version == api_version)
            
            page_count = json_page["count"]
            if count is None:
                count = page_count
            assert(page_count == count)

            page_data = json_page.get("data", [])
            if page_data:
                data.extend(page_data)

            # info:
            pages += 1
            rate = pages / duration_in_seconds

            if DEBUG:
                print(f"*** {url}: page [{pages}]: local records {len(page_data)}, total records {len(data)}/{count}, local duration: {elapsed_in_seconds}, total duration: {duration_in_seconds}, rate: {rate}, (max: {requests_per_second})")

            next_url = json_page.get("next")
            if next_url:
                complete = False
                next_params = None # params are already in next_url

                # respecter les limites de l'API
                if requests_per_second > 0:
                    if rate > requests_per_second:      
                        time.sleep(DELAY_BETWEEN_PAGES) # slow down
                        if DEBUG:                  
                            print(f"****** next : have to slow down")
            else:
                if DEBUG: 
                    print(f"****** next : done")

    return {"api_version": api_version, "etime": duration_in_seconds, "count": count, "data": data}

