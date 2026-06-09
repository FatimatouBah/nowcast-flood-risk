import os
import json
import mlflow
import logging
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
API_FAVICON_FILEPATH = "private/favicon-16.png"

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
    if not ((station_code in app_predictors_cache.keys()) and \
            (quantity_code in (app_predictors_cache[station_code]).keys())):
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
            "count": len(predictions), 
            "values": predictions
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

@app.get("/status")
async def get_status():
    """
    Get the health status of the application.

    Return { status }.
    """

    LOGGER.info("GET /status")
    return {"status": getattr(app, "status")}  

@app.get("/stations")
async def get_stations():
    """
    Get the list iof available stations.

    Return { api_version, count, values: [ { site, code, etc. } ] }.
    """

    LOGGER.info("GET /stations")
    try:
        stations = app_data_client.request_stations()
    except Exception as e:
        report_endpoint_exception(e, context="requesting stations")
    else:
        return stations

@app.get("/values/hixnj")
async def get_values_hixnj(station_code: str, from_date: dt.date, to_date: dt.date):
    """
    Get HIXnJ observations at a given station on a given date window.

    Mandatory query parameters : station_code, from_date, to_date.

    Return JSON content : { api_version, count, values: [ { ds, yobs } ] }.
    """

    LOGGER.info("GET /values/hixnj")
    try:
        observations = app_data_client.request_observations(
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

class HixnjStationPredictionFeatures(BaseModel):
    station_code: str
    from_date: dt.date
    to_date: dt.date

@app.post("/values/hixnj/predict") 
async def predict_values_hixnj(payload: HixnjStationPredictionFeatures):
    """
    Predict HIXnJ values at a given station on a given date window.

    Mandatory JSON body payload : { station_code, from_date, to_date }.

    Return JSON content : { api_version, count, values: [ { ds, yhat, yhat_lower, yhat_upper } ] }.
    """

    LOGGER.info("POST /values/hixnj/predict")
    try:
        predictor = fetch_station_predictor_from_cache(station_code=payload.station_code, quantity_code=QUANTITY_CODE_HIXNJ)
    except Exception as e:
        context = f"fetching {QUANTITY_CODE_HIXNJ} predictor for {{station: \"{payload.station_code}\"}}"
        report_endpoint_exception(e, context=context)
    else:
        try:
            predictions = do_predict_values(predictor, payload.from_date, payload.to_date)         
        except Exception as e:
            context = f"predicting {QUANTITY_CODE_HIXNJ} values for {{station: \"{payload.station_code}\", date_window: ({payload.from_date}, {payload.to_date})}}"
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
