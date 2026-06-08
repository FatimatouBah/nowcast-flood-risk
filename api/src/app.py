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
# MLFLOW setup
# -----------------------------------------------------------------------------

# ...

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

class HeightFeatures(BaseModel):
    StationCode: str
    DateTimeWindow: Tuple[dt.datetime,dt.datetime]

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get('/favicon.ico', include_in_schema=False)
async def get_favicon():
    return FileResponse(API_FAVICON_FILEPATH)

@app.get("/status")
async def get_status():
    LOGGER.info("GET /status")
    return {"status": True}

@app.get("/stations")
async def get_stations():
    LOGGER.info("GET /stations")
    stations = hb_client.request_stations()  # raise (http?) exception on dailure status
    return stations

@app.get("/values/heights")
async def get_heights(payload: HeightFeatures):
    LOGGER.info("GET /values/heights ")
    return {"count": 0, "data": []}

@app.post("/values/heights/predict")
async def predict_heights(payload: HeightFeatures):
    LOGGER.info("POST /values/heights/predict ")
    return {"count": 0, "data": []}

# ==============================================================================
# LOCAL TEST

if __name__ == "__main__":  
    import uvicorn
    HOST = "0.0.0.0"  # local host (always)
    PORT = 8000 # 7860 est le port standard pour Huggingface Spaces
    uvicorn.run(app, host=HOST, port=PORT)
