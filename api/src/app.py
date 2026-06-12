import os
import json
import time
import mlflow
import logging
import numpy as np
import pandas as pd
import datetime as dt
from pydantic import BaseModel
from typing import Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from dotenv import load_dotenv
load_dotenv()

# LOCAL COMPONENTS
from private.version import API_VERSION
from private.hubeau_client import HubeauClient

LOGGER = logging.getLogger(__name__)

API_TITLE = "Machine Learning Flood Forecasting API"
API_FAVICON_FILEPATH = "favicon.png"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def report_endpoint_exception(e, context):
    raise HTTPException(status_code=500, detail=f"exception caugth on {context}: {e}")

# -----------------------------------------------------------------------------
# MLFLOW setup
# -----------------------------------------------------------------------------
# Sur Hugging Face, ces variables sont lues depuis les "Secrets"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS")

# On force l'URI pour mlflow
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def build_mlflow_model_uri(quantity_code, station_code) -> Optional[str]:  # or None
    if quantity_code not in [QUANTITY_CODE_HIXNJ]:
        LOGGER.warning(f"There are no prediction models for quantity: {quantity_code}")
        return None
    
    # TODO : use a model dedicated to the given station.
    LOGGER.warning("Building MLflow model's URI : apply the same model for all stations...")

    # ex: models:/flood_forecast_model@baseline
    return f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"

# -----------------------------------------------------------------------------

class StationPredictor:
    def __init__(self, quantity_code, station_code, model):
        self.quantity_code = quantity_code
        self.station_code = station_code
        self.model = model # model dedicated to the given quantity and station

    def predict(self, from_date: dt.date, to_date: dt.date):
        def do_predict_(future_):
            df_ = self.model.predict(future_)
            assert(isinstance(df_, pd.DataFrame))
            fields_of_interest_ = ["ds", "yhat", "yhat_lower", "yhat_upper"]
            assert(all(foi in df_.columns for foi in fields_of_interest_))
            df_ = df_[fields_of_interest_]
            df_["ds"] = df_["ds"].dt.strftime('%Y-%m-%d')  # convert timestamps to formatted dates
            return df_
        
        # assume a prophet model : 
        #   - inputs  : DF("ds" [, ...])
        #   - outputs : DF("ds", "yhat", "yhat_lower", "yhat_upper" [, "trend", "daily", ...])

        df_future = pd.DataFrame({"ds": pd.date_range(start=from_date, end=to_date, freq='D')})
        df_values = do_predict_(df_future)

        output = json.loads(df_values.to_json(orient='records'))
        return output

STATION_CODE_KOGENHEIM = "A236003001"
QUANTITY_CODE_HIXNJ = "hixnj"

class HixnjStationPredictor(StationPredictor):
    def __init__(self, station_code, model):
        super().__init__(quantity_code=QUANTITY_CODE_HIXNJ, station_code=station_code, model=model)

app_predictors_cache = {}

# return an instance of StationPredictor or None
def load_prediction_model(station_code, quantity_code): 
    if quantity_code == QUANTITY_CODE_HIXNJ:
        model_uri = build_mlflow_model_uri(quantity_code=quantity_code, station_code=station_code)
        if model_uri is None:
            return None  # there are no models for the given quantity and station
        
    # DEBUG: print("****** on loading model at:", model_uri)
    model = mlflow.prophet.load_model(model_uri)
    # DEBUG: print(">>>>>> loaded:", model)
    return HixnjStationPredictor(station_code=station_code, model=model)

def fetch_station_predictor_from_cache(station_code, quantity_code):
    global app_predictors_cache
    if not ((station_code in app_predictors_cache.keys()) and (quantity_code in (app_predictors_cache[station_code]).keys())):
        predictor = load_prediction_model(station_code, quantity_code)
        if predictor is None:
            raise ValueError(f"cannot find <{quantity_code}> prediction model for station: <{station_code}>")
        app_predictors_cache[station_code] = dict(hixnj=predictor)

    predictor = app_predictors_cache[station_code][quantity_code]
    return predictor

def do_predict_values(predictor, from_date, to_date):
        predictions = predictor.predict(from_date, to_date)

        # return {"api_version", "count",  "values": [ {"ds", "yhat", ...} ] }
        return { 
            "api_version": API_VERSION, 
            "etime": 0.0,  # TODO
            "count": len(predictions), 
            "predictions": predictions
        }

# -----------------------------------------------------------------------------
# App's Hub'Eau client
# -----------------------------------------------------------------------------

app_data_client = HubeauClient()

# -----------------------------------------------------------------------------
# Load resources (models, etc.) on startup
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("loading resources on start-up...")
    try:
        # check the health of the system
        _ = fetch_station_predictor_from_cache(station_code=STATION_CODE_KOGENHEIM, quantity_code=QUANTITY_CODE_HIXNJ)
        LOGGER.info("Test model loaded successfully!")
        setattr(app, "status", True)
    except Exception as e:
        LOGGER.error(f"Failed to load test model: exception <{type(e)}> : {e}")
        setattr(app, "status", False)
    yield

# -----------------------------------------------------------------------------
# FastAPI setup
# -----------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan, version=API_VERSION, title=API_TITLE)

# health status:
setattr(app, "status", False)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get('/favicon.ico', include_in_schema=False)
async def get_favicon():
    return FileResponse(API_FAVICON_FILEPATH)

@app.get("/ping")
async def ping():
    """
    Get the health status of the application.

    Return { status }.
    """

    LOGGER.info("GET /ping")
    return {"status": getattr(app, "status")}  

def _is_ts_cache_outdated(cache, now) -> tuple[bool, dict]:
    # DEBUG : print("...... inside _is_ts_cache_outdated: 1")
    APP_CACHE_TTL = 300  # 5 minutes
    if not isinstance(cache, dict): 
        new_cache = dict(ts=0)
        return True, new_cache
    if "ts" not in cache.keys(): 
        cache["ts"] = 0
        return True, cache
    assert(isinstance(cache, dict) and ("ts" in cache.keys()))
    outdated = now > cache["ts"] + APP_CACHE_TTL
    return outdated, cache

def _get_station_quantity_observations_cache(station_code, quantity_code, inout_caches_map = None):
    if not isinstance(inout_caches_map, dict): 
        inout_caches_map = dict()
    if station_code not in inout_caches_map.keys():
        inout_caches_map[station_code] = dict()
    station_caches = inout_caches_map[station_code]
    if quantity_code not in station_caches.keys():
        station_caches[quantity_code] = dict(ts=0)
    station_quantity_cache = station_caches[quantity_code]
    return station_quantity_cache

app_stations_cache = {}

@app.get("/stations")
async def get_stations():
    """
    Get the list iof available stations.

    Return { api_version, etime, count, stations: [ { site, code, etc. } ] }.
    """
    LOGGER.info("GET /stations")
    try:
        global app_stations_cache
        global app_data_client
        now = time.time()
        # DEBUG : print("****** calling _is_ts_cache_outdated:")
        outdated, app_stations_cache = _is_ts_cache_outdated(cache=app_stations_cache, now=now)
        # DEBUG : print("****** _is_ts_cache_outdated called")
        assert(isinstance(app_stations_cache, dict))
        if outdated:
            app_stations_cache["data"] = app_data_client.request_stations()
            app_stations_cache["ts"] = now
    except Exception as e:
        report_endpoint_exception(e, context="requesting stations")
    else:
        data_cache = app_stations_cache["data"]
        return data_cache
    
@app.get("/station/hixnj/dates")
async def get_station_hixnj_dates(station_code: str):
    """
    Get the dates window of the available HIXnJ observations at the given station.

    Mandatory query parameters : station_code.

    Return JSON content : { api_version, etime, dates: { lower, upper } }.
    """

    LOGGER.info("GET /station/hixnj/dates")
    try:
        global app_data_client
        dates = app_data_client.request_station_dates(
            quantity_code=QUANTITY_CODE_HIXNJ,
            station_code=station_code
        )    
    except Exception as e:
        context = f"requesting dates for available {QUANTITY_CODE_HIXNJ} observations for {{station: {station_code}}}"
        report_endpoint_exception(e, context=context)
    else:    
        return dates

@app.get("/station/hixnj/stats")
async def get_station_hixnj_stats(station_code: str):
    """
    Get statistics of the available HIXnJ observations at the given station (min, max, q98, etc.).

    Mandatory query parameters : station_code.

    Return JSON content : { api_version, etime, dates: { lower, upper }, stats: { min, max, ..., q98 } }.
    """

    LOGGER.info("GET /station/hixnj/stats")
    try:
        global app_data_client
        thresholds = app_data_client.request_station_stats(
            quantity_code=QUANTITY_CODE_HIXNJ,
            station_code=station_code
        )    
    except Exception as e:
        context = f"requesting {QUANTITY_CODE_HIXNJ} thresholds for {{station: {station_code}}}"
        report_endpoint_exception(e, context=context)
    else:
        return thresholds

app_stations_observations_cache = None

@app.get("/station/hixnj/observations")
async def get_station_hixnj_observations(station_code: str, from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None):
    """
    Get qualified HIXnJ observations at the given station, optionally on the given dates window.

    Mandatory query parameters : station_code.

    Return JSON content : { api_version, etime, count, dates, stats, observations: [ { ds, yobs } ] }.
    """
    
    datetypes = (dt.date, )

    def is_date_inside_window_(date_, lower_date_, upper_date_):
        return ((lower_date_ is None) or (date_ >= lower_date_)) and ((upper_date_ is None) or (date_ <= upper_date_))
        
    def get_station_hixnj_observations_cache_(station_code_, inout_caches_map_):
        return _get_station_quantity_observations_cache(station_code=station_code_, quantity_code=QUANTITY_CODE_HIXNJ, inout_caches_map=inout_caches_map_)

    LOGGER.info("GET /station/hixnj/observations")
    try:
        global app_stations_observations_cache
        global app_data_client
        now = time.time()
        observations_cache = get_station_hixnj_observations_cache_(station_code, app_stations_observations_cache)
        outdated, observations_cache =  _is_ts_cache_outdated(cache=observations_cache, now=now)
        assert(isinstance(observations_cache, dict))
        if outdated:
            observations_cache["data"] = app_data_client.request_station_hixnj_observations(station_code=station_code)  # get all observations
            observations_cache["ts"] = now
    except Exception as exc:
        context = f"requesting {QUANTITY_CODE_HIXNJ} observations for {{station: {station_code}, from_date: {from_date}, to_date: {to_date}}}"
        report_endpoint_exception(exc, context=context)
    else:
        data_cache = observations_cache["data"]
        if (from_date is None and to_date is None):
            return data_cache

        assert(isinstance(from_date, datetypes) or isinstance(to_date, datetypes))
        data_cache2 = {**data_cache}
        data_cache2["observations"] = [obs for obs in data_cache2["observations"] if is_date_inside_window_(dt.date.fromisoformat(obs["ds"]), from_date, to_date)]
        return data_cache2

class HixnjStationPredictionFeatures(BaseModel):
    station_code: str
    from_date: dt.date
    to_date: dt.date

@app.post("/station/hixnj/predict") 
async def predict_station_hixnj(payload: HixnjStationPredictionFeatures):
    """
    Predict HIXnJ values at the given station on the given dates window.

    Mandatory JSON body payload : { station_code, from_date, to_date }.

    Return JSON content : { api_version, etime, count, predictions: [ { ds, yhat, yhat_lower, yhat_upper } ] }.
    """

    LOGGER.info("POST /station/hixnj/predict")
    try:
        predictor = fetch_station_predictor_from_cache(station_code=payload.station_code, quantity_code=QUANTITY_CODE_HIXNJ)
    except Exception as e:
        context = f"fetching {QUANTITY_CODE_HIXNJ} predictor for {{station: {payload.station_code}}}"
        report_endpoint_exception(e, context=context)
    else:
        try:
            predictions = do_predict_values(predictor, payload.from_date, payload.to_date)         
        except Exception as e:
            context = f"predicting {QUANTITY_CODE_HIXNJ} values for {{station: {payload.station_code}, date_window: ({payload.from_date}, {payload.to_date})}}"
            report_endpoint_exception(e, context=context)
        else:
            return predictions

# ==============================================================================
# LOCAL TEST

if __name__ == "__main__":  
    import uvicorn
    HOST = "0.0.0.0"  # local host (always)
    PORT = 8000 # 7860 est le port standard pour Huggingface Spaces
    uvicorn.run(app, host=HOST, port=PORT)
