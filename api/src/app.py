import logging
import datetime as dt
from pydantic import BaseModel
from typing import Tuple, List, Union, Literal

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
API_FAVICON_FILEPATH = "favicon-16.png"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def report_endpoint_exception(e, context):
    raise HTTPException(status_code=500, detail=f"exception caugth on {context}: {e}")

# -----------------------------------------------------------------------------
# MLFLOW setup
# -----------------------------------------------------------------------------

# TMP :

def date_range(d1, d2):
    current = d1
    while current <= d2:
        yield current
        current += dt.timedelta(days=1)

class StationPredictor:
    def __init__(self, quantity_code, station_code):
        self.quantity_code = quantity_code
        self.station_code = station_code

    def predict(self, from_date: dt.date, to_date: dt.date):
        def predict_fake_(ds_):
            return 0.0
    
        dated_values = [ {"ds": ds, "y": predict_fake_(ds)} for ds in date_range(from_date, to_date)]
        return dated_values

QUANTITY_CODE_HIXNJ = "hixnj"

class HixnjStationPredictor(StationPredictor):
    def __init__(self, station_code):
        super().__init__(quantity_code=QUANTITY_CODE_HIXNJ, station_code=station_code)

predictors_cache = {
    "A236003001": {
        QUANTITY_CODE_HIXNJ: HixnjStationPredictor(station_code="A236003001")
    }
}

# return an instance of StationPredictor or 
def load_prediction_model(station_code, quantity_code): 
    return HixnjStationPredictor(station_code=station_code) if quantity_code == QUANTITY_CODE_HIXNJ else None

def fetch_station_predictor_from_cache(station_code, quantity_code):
    if not ((station_code in predictors_cache.keys()) and \
            (quantity_code in (predictors_cache[station_code]).keys())):
        predictor = load_prediction_model(station_code, quantity_code)
        if predictor is None:
            raise ValueError(f"cannot find <{quantity_code}> prediction model for station: <{station_code}>")
        predictors_cache[station_code] = dict(hixnj=predictor)

    predictor = predictors_cache[station_code][quantity_code]
    return predictor

def do_predict_values(predictor, from_date, to_date):
        predictions = predictor.predict(from_date, to_date)

        # return {"api_version", "count",  "values": [ {"ds", "y"} ] }
        return { 
            "api_version": API_VERSION, 
            "count": len(predictions), 
            "values": predictions
        }

# -----------------------------------------------------------------------------
# HUBEAU client
# -----------------------------------------------------------------------------

hb_client = HubeauClient()

# -----------------------------------------------------------------------------
# Load resources (models, etc.) on startup
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("loading resources on start-up...")
    yield

# -----------------------------------------------------------------------------
# FastAPI Setup
# -----------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan, version=API_VERSION, title=API_TITLE)

class HixnjStationPredictionFeatures(BaseModel):
    station_code: str
    from_date: dt.date
    to_date: dt.date

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get('/favicon.ico', include_in_schema=False)
async def get_favicon():
    return FileResponse(API_FAVICON_FILEPATH)

@app.get("/status")
async def get_status():
    """
    Get the health status of the application.

    Return { status }.
    """

    LOGGER.info("GET /status")
    return {"status": True}  

@app.get("/stations")
async def get_stations():
    """
    Get the list iof available stations.

    Return { api_version, count, values: [ { site, code, etc. } ] }.
    """

    LOGGER.info("GET /stations")
    try:
        stations = hb_client.request_stations()
    except Exception as e:
        report_endpoint_exception(e, context="requesting stations")
    else:
        return stations

@app.get("/values/hixnj")
async def get_values_hixnj(station_code: str, from_date: dt.date, to_date: dt.date):
    """
    Get HIXnJ observations at a given station on a given date window.

    Mandatory query parameters : station_code, from_date, to_date.

    Return { api_version, count, values: [ { ds, y } ] }.
    """

    LOGGER.info("GET /values/hixnj")
    try:
        observations = hb_client.request_observations(
            quantity_code=QUANTITY_CODE_HIXNJ, 
            station_code=station_code,
            from_date=from_date, 
            to_date=to_date
        )
    except Exception as e:
        context = f"requesting {QUANTITY_CODE_HIXNJ} values for {{station: \"{station_code}\", from_date: {from_date}, to_date: {to_date}}}"
        report_endpoint_exception(e, context=context)
    else:
        return observations

@app.post("/values/hixnj/predict") 
async def predict_values_hixnj(payload: HixnjStationPredictionFeatures):
    """
    Predict HIXnJ values at a given station on a given date window.

    Mandatory body payload : { station_code, from_date, to_date }.
    
    Return { api_version, count, values: [ { ds, y } ] }.
    """

    LOGGER.info("POST /values/hixnj/predict")
    try:
        predictor = fetch_station_predictor_from_cache(station_code=payload.station_code, quantity_code=QUANTITY_CODE_HIXNJ)
    except Exception as e:
        context = f"fetching {QUANTITY_CODE_HIXNJ} predictor for {{station: \"{payload.station_code}\"}}"
        report_endpoint_exception(e, context=context)
    else:
        try:
            predictions = do_predict_values(predictor, payload.date_window)         
        except Exception as e:
            context = f"predicting {QUANTITY_CODE_HIXNJ} predictor for {{station: \"{payload.station_code}\", date_window: ({payload.from_date}, {payload.to_date})}}"
            report_endpoint_exception(e, context=f"context")
        else:
            return predictions

# ==============================================================================
# LOCAL TEST

if __name__ == "__main__":  
    import uvicorn
    HOST = "0.0.0.0"  # local host (always)
    PORT = 8000 # 7860 est le port standard pour Huggingface Spaces
    uvicorn.run(app, host=HOST, port=PORT)
