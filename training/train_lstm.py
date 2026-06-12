# -*- coding=utf-8 -*-
from pathlib import Path
import os
import shutil
from datetime import datetime, date
import time  # Import time to measure the duration of the training process
from typing import Any, cast
import requests  # Import requests to call the Hub'Eau API.

# Import the load_dotenv function to read key-value pairs from a .env file into the OS environment
from dotenv import load_dotenv

# Import argparse to parse command-line arguments for the script, allowing for flexible configuration of model parameters and training options when running the script from the terminal.
import argparse

import warnings  # Import warnings to silence noisy logs for a cleaner UI.

# ============================================================
# Necessary modules for data manipulation
# ============================================================
# Import de NumPy pour les calculs numériques
import numpy as np
# Import de pandas pour manipuler les DataFrames
import pandas as pd

# ============================================================
# Necessary modules for Visualization
# ============================================================
# Import de matplotlib pour les graphiques
import matplotlib.pyplot as plt
# Import de seaborn pour des visualisations plus propres
import seaborn as sns

# Import plotly for interactive visualizations
from plotly import express as px
from plotly import graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

# ============================================================
# Necessary modules for MLFlow
# ============================================================
# Import MLflow and its Prophet integration to log the model training process, parameters, metrics, and artifacts to a tracking server (Neon DB) for better experiment management and reproducibility.
import mlflow
import mlflow.sklearn
# import mlflow.prophet
from mlflow import prophet as mlflow_prophet
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient  # Import MlflowClient to interact with the MLflow tracking server

# Import MLflow TensorFlow integration to log TensorFlow/Keras models and their training process, allowing us to track the performance of the LSTM model and compare it with other models in a consistent way within the MLflow ecosystem.
import mlflow.tensorflow

# Import joblib and pickle for model serialization
import pickle
import joblib

# ============================================================
# Necessary modules for Scikit-learn
# ============================================================
# Import train_test_split and TimeSeriesSplit for splitting the data, and various metrics for evaluating the model"s performance
from sklearn.model_selection import train_test_split, TimeSeriesSplit
# Import des métriques sklearn pour l'évaluation
from sklearn.metrics import  (r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error)

# ============================================================
# Necessary modules for Prophet
# ============================================================
# Import Prophet and its diagnostics tools for time series forecasting and model evaluation, allowing us to build a forecasting model for the water height variable and assess its performance using cross-validation techniques.
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# ============================================================
# Necessary modules for LSTM 
# ============================================================

# Import du scaler MinMax pour normaliser les données temporelles
from sklearn.preprocessing import MinMaxScaler
# Import de la fonction pour créer des séquences temporelles à partir des données brutes, ce qui est nécessaire pour entraîner un modèle LSTM qui prend en compte les dépendances temporelles dans les données.
import tensorflow as tf
# Import TensorFlow / Keras pour le modèle LSTM
from tensorflow.keras.models import Sequential
# Import des couches réseau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
# Import des callbacks pour éviter le sur-apprentissage
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")  # Hide non-critical warnings to keep output simple.

# specify to use GPU or CPU device
# Check if GPU is available and set the device accordingly
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    device_name = '/device:CPU:0'
print(f"Using device: {device_name}")

# set CPU or GPU device for TensorFlow operations
tf.config.set_visible_devices([], 'GPU') if device_name == '/device:CPU:0' else tf.config.set_visible_devices([], 'CPU')


# URL base API HubEau hydrométrie
API_BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"

# Sites de mesures hydrométries pour l'entraînement du modèle de prévision de risque d'inondation
SITES = {
    # Site principal (Kogenheim)
    "Kogenheim":  {
        "code" : "A2360030",
        "site": "main",
        "river": "ill",
        "region": "grand-est",
		"stations": [
			{"code": "A236003001", "municipality" :"Kogenheim"}
		],
        "map": [
            {"latitude": "48.3370", "longitude": "7.5466"},
        ]
	},
    "Colmar-1":     {
        "code" : "A1580201", 
        "site": "main",
        "river": "launch",
        "region": "grand-est",
		"stations": [
            {"code": "A158020101", "municipality" :"Colmar"}, 
        ],
        "map": [
            {"latitude": "48.1025", "longitude": "7.3850"},
        ]
	},
    "Colmar-2":     {
        "code" : "A1610030", 
        "site": "main",
        "river": "ill",
        "region": "grand-est",
		"stations": [
            {"code": "A161003001", "municipality" :"Colmar"}, 
        ],
        "map": [
            {"latitude": "48.0887", "longitude": "7.4438"},
        ]
	},
    "Colmar-3":     {
        "code" : "A2220001", 
        "site": "main",
        "river": "fetch",
        "region": "grand-est",
		"stations": [
            {"code": "A222000101", "municipality" :"Colmar"}, 
        ],
        "map": [
            {"latitude": "48.1617", "longitude": "7.4485"},
        ]
	},
    "Selestat":   {
        "code" : "A2350200",
        "site": "main",
        "river": "giessen",
        "region": "grand-est",
		"stations": [
			{"code": "A235020001", "municipality" :"Selestat"},
			{"code": "A235020002", "municipality" :"Selestat"},
			{"code": "A235020003", "municipality" :"Selestat"}
		], 
        "map": [
            {"latitude": "48.2718", "longitude": "7.4598"},
            {"latitude": "48.2728", "longitude": "7.4566"},
            {"latitude": "48.2381", "longitude": "7.3915"}
        ]
	},
    "Ostheim":    {
        "code" : "A2140100",
        "site": "main",
        "river": "fetch",
        "region": "grand-est",
		"stations": [
			{"code": "A214010001", "municipality" :"Ostheim"}
		], 
        "map": [
            {"latitude": "48.2085", "longitude": "7.3976"},
        ]
	},

    # Site secondaire (Waltenheim)
    
    "Waltenheim": {
        "code" : "A3480200",
        "site": "secondary",
        "river": "zorn",
        "region": "grand-est",
		"stations": [
			{"code": "A348020001", "municipality" :"Waltenheim-sur-Zorm"}
		], 
        "map": [
            {"latitude": "48.7496", "longitude": "7.6337"},
        ]
	},
    "Oberhof":    {
        "code" : "A3430210",
        "site": "secondary",
        "river": "zinsel-sud",
        "region": "grand-est",
		"stations": [
			{"code": "A343021001", "municipality" :"Eckartswiller"}
		], 
        "map": [
            {"latitude": "48.7996", "longitude": "7.3105"},
        ]
	},
    "Saverne":    { 
        "code" : "A3410200",
        "site": "secondary",
        "river": "zorn",
        "region": "grand-est",
		"stations": [
			{"code": "A341020001", "municipality" :"Saverne"}
		],
        "map": [
            {"latitude": "48.7380", "longitude": "7.3747"},
        ]
	},
}

AVAILABLE_MEASURES = ["QmnJ", "QmM", "HIXM", "HIXnJ", "QINM", "QINnJ", "QixM", "QIXnJ"]  # Define known Hub'Eau measures.



###################################################################################################################################
#                       DONT TOUCH BELOW UNLESS YOU KNOW WHAT YOU ARE DOING
# THIS IS THE STANDARD SETUP FOR TRAINING AND EVALUATING THE PROPHET MODEL, LOGGING TO MLflow, AND SAVING ARTIFACTS LOCALLY
###################################################################################################################################




# =======================================================================
# ENVIRONMENT VARIABLES SETUP
# =======================================================================
# Execute the load_dotenv function to populate os.environ with variables from the .env file
load_dotenv()

# Set environment variable to prevent MLflow from recording all environment variables during model logging 
# (optional, can help reduce clutter in the logged model metadata)
os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "False"
# Set environment variable for MLflow tracking URI to use local file-based storage (optional, defaults to local file-based storage)
# Retrieve the MLFLOW_TRACKING_URI (Neon DB URL) from the environment variables
# os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# Retrieve the MLFLOW_REGISTRY_URI from the environment variables, or reuse the tracking URI when both services share the same endpoint.
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_TRACKING_URI")
# Retrieve the MLFLOW_ARTIFACT_URI (S3 Bucket path) from the environment variables
MLFLOW_ARTIFACT_URI = os.getenv("ARTIFACT_ROOT")
# Retrieve the BACKEND_STORE_URI (Neon DB URL) from the environment variables
BACKEND_STORE_URI = os.getenv("BACKEND_STORE_URI")

# # Print the effective tracking URI at startup to avoid ambiguity
# print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
# print(f"MLflow Registry URI: {mlflow.get_registry_uri()}")
# print(f"MLflow Artifact URI setting: {MLFLOW_ARTIFACT_URI}")



# get the working folder containing train.py
WorkDir = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = WorkDir.parent
print(f"Current working directory: {WorkDir}")


Current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Current time: {Current_time}")

# Create artifact directory (output file) to save the results of the model training and evaluation
local_artifact_dir = WorkDir / f"artifact_outputs_{Current_time}"
if not local_artifact_dir.exists():
    local_artifact_dir.mkdir(parents=True)



def fetch_obs_elab(code_station: str, start_date: str, end_date: str, measure: str) -> pd.DataFrame:
    """Fetch elaborated hydrometric observations from Hub'Eau and return a DataFrame."""  # Function purpose.
    endpoint = f"{API_BASE_URL}/obs_elab"  # Build the API endpoint URL.
    params = {  # Build request query parameters.
        "code_entite": code_station,  # Set station code.
        "date_debut_obs_elab": start_date,  # Set start date filter.
        "date_fin_obs_elab": end_date,  # Set end date filter.
        "size": 20000,  # Request maximum records per page.
        "format": "json",  # Request JSON response format.
        "grandeur_hydro_elab": measure,  # Select hydrometric measure.
    }

    try:  # Protect API request against connection issues.
        response = requests.get(endpoint, params=params, timeout=45)  # Send HTTP GET with timeout.
        response.raise_for_status()  # Raise error on non-2xx HTTP status.
        payload = response.json()  # Parse response body as JSON.
    except Exception as exc:  # Catch network, HTTP, or parsing errors.
        print(f"API error: {exc}")  # Show readable error in Streamlit.
        return pd.DataFrame()  # Return empty DataFrame to stop pipeline safely.

    rows = payload.get("data", []) if isinstance(payload, dict) else []  # Extract data rows from payload.
    df = pd.DataFrame(rows)  # Convert list of records to DataFrame.
    return df  # Return raw DataFrame.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MLflow + scikit-learn training script")
    parser.add_argument("--site_name", type=str, default="Kogenheim", help="Site name")
    parser.add_argument("--code_site", type=str, default="A2360030", help="Site code")
    parser.add_argument("--code_station", type=str, default="A236003001", help="Station code")
    parser.add_argument("--site_measure", type=str, default="HIXnJ", help="Site measure")
    parser.add_argument("--start_date", type=str, default="2007-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default=date.today(), help="End date")
    
    parser.add_argument("--seasonality", type=bool, default=False, help="Include seasonality components (True or False)")
    
    parser.add_argument("--changepoint_prior_scale", type=float, default=0.05, help="Changepoint prior scale")
    parser.add_argument("--seasonality_prior_scale", type=float, default=10, help="Seasonality prior scale")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--ncpus", type=int, default=1, help="Number of CPUs")
    # parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of test set")
    return parser.parse_args()


def export_model_with_pickle(model: Prophet, output_path: str) -> None:
    # Export the trained Prophet model to a local binary file so the model can be reused outside MLflow as well.
    with open(output_path, "wb") as model_file:
        pickle.dump(model, model_file)


def export_model_with_joblib(model: Prophet, output_path: str) -> None:
    # Save the trained model to a file using joblib
    with open(output_path, "wb") as model_file:
        joblib.dump(model, model_file)




# *====================================================================
# MODEL GLOBAL PARAMETERS AND SETTINGS
# *====================================================================

# Define the column names for the datetime and target variable in the raw dataset
DATETIME_COLUMN_NAME = "date_obs_elab"
TARGET_COLUMN_NAME = "resultat_obs_elab"



def split_train_test(series: pd.DataFrame, train_ratio: float = 0.8, split_by_ratio=True, SPLIT_DATE="2024-01-01") -> tuple[pd.DataFrame, pd.DataFrame, str|float]:
    """Split time series into chronological train and test sets."""  # Function purpose.
    if len(series) < 20:  # Enforce minimum length for meaningful evaluation.
        return None, None, SPLIT_DATE  # Return invalid split markers.
    
    if split_by_ratio:
        train_size = max(int(len(series) * train_ratio), 10)  # Compute train size with a small lower bound.
        train_size = min(train_size, len(series) - 5)  # Keep at least five points for test.
        train_df = series.iloc[:train_size].copy()  # Create train subset.
        test_df = series.iloc[train_size:].copy()  # Create test subset.
        # Get the last date of the training set for reference
        SPLIT_DATE = train_df.index[-1]  # get the last date of the training set as the split date
        return train_df, test_df, SPLIT_DATE  # Return split datasets and the split date used for reference.
    else:
        mask_selection = series.index < SPLIT_DATE
        train_df = series[mask_selection].copy()
        test_df = series[~mask_selection].copy()
        ratio = len(train_df) / len(series)  # Calculate the actual ratio of the split for reference
        return train_df, test_df, ratio  # Return split datasets and the actual ratio of the split for reference


def prepare_series(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a clean daily time series with Prophet-compatible columns ds and y."""  # Function purpose.
    required_cols = {DATETIME_COLUMN_NAME, TARGET_COLUMN_NAME}  # Define required source columns.
    if df.empty or not required_cols.issubset(df.columns):  # Validate DataFrame availability and schema.
        return pd.DataFrame()  # Return empty DataFrame when input is unusable.
    
    # I.1. Select the relevant columns and rename them to "ds" and "y"
    series = df[[DATETIME_COLUMN_NAME, TARGET_COLUMN_NAME]].copy()
    # series = df[["date_obs_elab", "resultat_obs_elab"]].copy()  # Keep only date and target columns.
    series = series.rename(columns={DATETIME_COLUMN_NAME: "ds", TARGET_COLUMN_NAME: "y"})
    # series.rename(columns={"date_obs_elab": "ds", "resultat_obs_elab": "y"}, inplace=True)  # Rename columns.
    
    # I.2. Ensure the "ds" column is a proper datetime type.
    # Note: Prophet often prefers timezone-naive datetimes, so we remove the timezone if it exists.
    # series["ds"] = pd.to_datetime(series["ds"]).dt.tz_localize(None)
    series["ds"] = pd.to_datetime(series["ds"], errors="coerce")  # Convert date column to datetime.
    
    # I.3. Ensure "y" is numeric
    # clean_data_df["y"] = pd.to_numeric(clean_data_df["y"], errors="coerce")
    series["y"] = pd.to_numeric(series["y"], errors="coerce")  # Convert target column to numeric.

    # # Optional: drop any rows with NaN values in "y" as they can cause issues
    # clean_data_df = clean_data_df.dropna(subset=["y"])
    series.dropna(subset=["ds", "y"], inplace=True)  # Drop invalid rows.
    
    # I.4. Sort the series by date to ensure chronological order, which is important for time series analysis and modeling.
    series.sort_values("ds", inplace=True)  # Sort rows by date.
    
    series.drop_duplicates(subset="ds", keep="last", inplace=True)  # Keep latest value for duplicate dates.
    series.reset_index(drop=True, inplace=True)  # Reset index after cleaning.
    return series  # Return clean series.


def plot_time_series_with_plotly(clean_data_df: pd.DataFrame, STATION_MEASURE: str, TITLE_POSTFIX_NAME: str):
    # I.4. Time Series With Range Slider
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=clean_data_df["ds"], y=clean_data_df["y"], mode="lines", name=f"Water Height ({STATION_MEASURE})"))
    # fig1.add_trace(go.Scatter(gold_ts, x="Date", y="Close", mode="lines", name="Close"))
    # fig1.add_trace(go.Scatter(gold_ts, x="Date", y=gold_ts.columns, mode="lines", name="Close"))
    # Customize the current trace
    fig1.update_traces(line_color="blue", name=f"Water Height ({STATION_MEASURE})", showlegend=True)
    # add horizontal line for flood threshold (example: 2.86) in April 1983
    fig1.add_hline(y=2.86*1000, line_dash="dash", line_color="red", annotation_text="Flood Threshold (2.86) in April 1983", annotation_position="top left")
    # add horizontal line for flood threshold (example: 2.77) in February 1990
    fig1.add_hline(y=2.77*1000, line_dash="dash", line_color="cyan", annotation_text="Flood Threshold (2.77) in February 1990", annotation_position="bottom left")
    # add horizontal line for flood threshold (example: 2.58) in January 2018
    fig1.add_hline(y=2.58*1000, line_dash="dash", line_color="green", annotation_text="Flood Threshold (2.58) in January 2018", annotation_position="bottom left")
    # Customize the layout of the figure
    fig1.update_layout(
        title=f"{TITLE_POSTFIX_NAME} : Water Height Over Time",
        title_x=0.5,
        title_font=dict(size=24, color="black", family="Arial", weight="bold"),
        xaxis_title="Datetime", 
        yaxis_title=f"Water Height ({STATION_MEASURE})",
        legend_title="Legend",
    #   showlegend=True,
    #   legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)", bordercolor="black", borderwidth=1),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"), # 1 day
                    dict(count=7, label="7d", step="day", stepmode="backward"), # 7 days
                    dict(count=1, label="1m", step="month", stepmode="backward"), # 1 month
                    dict(count=3, label="3m", step="month", stepmode="backward"), # 3 months
                    dict(count=6, label="6m", step="month", stepmode="backward"), # 6 months
                    dict(count=1, label="YTD", step="year", stepmode="todate"), # Year to date
                    dict(count=1, label="1y", step="year", stepmode="backward"), # 1 year
                    dict(step="all") # All data 
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        # Size of the figure
        width=1200, height=600
    )
    return fig1


def plot_train_test_split(train_fbp: pd.DataFrame, test_fbp: pd.DataFrame, SPLIT_DATE: str, STATION_MEASURE: str, TITLE_POSTFIX_NAME: str):
    # Visualize the train/test split using plotly
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train_fbp["ds"], y=train_fbp["y"], mode="lines", name="Train Set"))
    fig2.add_trace(go.Scatter(x=test_fbp["ds"], y=test_fbp["y"], mode="lines", name="Test Set"))
    fig2.add_vline(x=SPLIT_DATE, line_dash="dash", line_color="black")
    # fig2.add_vline(x=SPLIT_DATE, line_dash="dash", line_color="red", annotation_text="Train-Test split line", annotation_position="top left")
    # Customize the layout of the figure
    fig2.update_layout(title=f"Data Train/Test Split - Water Height Over Time - {TITLE_POSTFIX_NAME}",
                    title_x=0.5,
                    title_font=dict(size=24, color="black", family="Arial", weight="bold"),
                    xaxis_title="Datetime", 
                    yaxis_title=f"Water Height ({STATION_MEASURE})",
                    legend_title="Legend",
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)", bordercolor="black", borderwidth=1),
                    # Size of the figure
                    width=1200, height=600)
    return fig2


def plot_prophet_components(prophet_model: Prophet, test_fcst: pd.DataFrame, local_artifact_dir: Path, FILE_POSTFIX_NAME: str, TITLE_POSTFIX_NAME: str) -> None:
    # Plot the components of the model
    fig3 = prophet_model.plot_components(test_fcst)
    # save the figure as an HTML and png file
    fig3.savefig(local_artifact_dir / f"03_prophet_components_plot_{FILE_POSTFIX_NAME}.png")
    # fig3.show()
    
    # Plot the components of the model using plotly
    fig4 = make_subplots(rows=4, cols=1, subplot_titles=["Trend", "Weekly Seasonality", "Yearly Seasonality", "Daily Seasonality"])
    fig4.add_trace(go.Scatter(x=test_fcst["ds"], y=test_fcst["trend"], mode="lines", name="Trend"), row=1, col=1)
    fig4.add_trace(go.Scatter(x=test_fcst["ds"], y=test_fcst["weekly"], mode="lines", name="Weekly Seasonality"), row=2, col=1)
    fig4.add_trace(go.Scatter(x=test_fcst["ds"], y=test_fcst["yearly"], mode="lines", name="Yearly Seasonality"), row=3, col=1)
    fig4.add_trace(go.Scatter(x=test_fcst["ds"], y=test_fcst["daily"], mode="lines", name="Daily Seasonality"), row=4, col=1)
    # Customize the layout of the figure
    fig4.update_layout(title=f"Prophet Model Components - Water Height Over Time - {TITLE_POSTFIX_NAME}",
                    title_x=0.5,
                        title_font=dict(size=24, color="black", family="Arial", weight="bold"),
                        xaxis_title="Datetime", 
                        yaxis_title="Component Value",
                        legend_title="Legend",
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)", bordercolor="black", borderwidth=1),
                        height=900, width=1200)
    return fig4


def plot_forecast_with_plotly(prophet_model, forecast: pd.DataFrame, STATION_MEASURE: str, TITLE_POSTFIX_NAME: str):
    # Plot the forecasted results using plotly
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecasted Water Height (yhat)"))
    fig6.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Confidence Interval (yhat_lower)", line=dict(dash="dash", color="red")))
    fig6.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Confidence Interval (yhat_upper)", line=dict(dash="dash", color="green")))
    # Customize the layout of the figure
    fig6.update_layout(title=f"Prophet Forecast - Water {STATION_MEASURE} Over Time - {TITLE_POSTFIX_NAME}",
                    title_x=0.5,
                        title_font=dict(size=24, color="black", family="Arial", weight="bold"),
                        xaxis_title="Datetime", 
                        yaxis_title=f"Water {STATION_MEASURE}",
                        legend_title="Legend",
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)", bordercolor="black", borderwidth=1),
                        height=900, width=1200)
    return fig6


def compute_prophet_metrics(test_fbp: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    # We now measure how accurate the forecast is. If there’s no overlap in dates, it warns you instead of crashing.

    #VI.1. Align predictions with actuals
    # Join actual water height with model predictions on matching dates to compare them directly.

    # Join actual water height with model predictions on matching dates to compare them directly.
    # We merge the test dataframe (which contains the actual water height for the test period) with the forecast dataframe (which contains the predicted values for the same period) on the "ds" column, 
    # which represents the dates.
    # The merge is done using an inner join, which means that only the rows with matching dates in both dataframes will be included in the resulting merged dataframe.
    merged = pd.merge(test_fbp, forecast[["ds", "yhat"]], on="ds", how="inner")

    # VI.2. Compute (r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error)
    # We now measure how accurate the forecast is. If there’s no overlap in dates, it warns you instead of crashing.

    mape = float("nan")
    r2 = float("nan")
    mse = float("nan")
    rmse = float("nan")
    mae = float("nan")
    prophet_metrics = {}

    if merged.empty:
        # If the merged dataframe is empty, it means there are no overlapping dates between the test set and the forecasted values, 
        # so we print a message indicating that.
        print("No overlapping dates between forecast and test set.")
    else:
        y_actual, y_predicted = merged["y"], merged["yhat"]
        # If the merged dataframe is not empty, it means there are overlapping dates between the test set and the forecasted values, 
        # calculate the Mean Absolute Percentage Error (MAPE) between the actual values (y) and the predicted values (yhat).
        mape = mean_absolute_percentage_error(y_actual, y_predicted)
        # calculate the r^2 score
        r2 = r2_score(y_actual, y_predicted)
        # calculate the mean squared error (MSE)
        mse = mean_squared_error(y_actual, y_predicted)
        # calculate the root mean squared error (RMSE)
        rmse = root_mean_squared_error(y_actual, y_predicted) # = np.sqrt(mse)
        # calculate the mean absolute error (MAE)
        mae = mean_absolute_error(y_actual, y_predicted)
        
        #save prophet metrics into a dictionary for later logging to MLflow
        prophet_metrics = {"std_mape": mape,
                            "std_r2": r2,
                            "std_mse": mse,
                            "std_rmse": rmse,
                            "std_mae": mae}
    return prophet_metrics


def plot_seasonal_components(test_fcst: pd.DataFrame, TITLE_POSTFIX_NAME: str) -> go.Figure:
    # Use plotly to plot seasonal components
    fig9 = make_subplots(rows=4, cols=1, subplot_titles=["Trend", "Weekly Seasonality", "Yearly Seasonality", "Daily Seasonality"])
    fig9.add_trace(go.Scatter(x=test_fcst["ds"], y=test_fcst["trend"], mode="lines", name="Trend"), row=1, col=1)
    fig9.add_trace(go.Scatter(x=test_fcst["ds"], y=test_fcst["weekly"], mode="lines", name="Weekly Seasonality"), row=2, col=1)
    fig9.add_trace(go.Scatter(x=test_fcst["ds"], y=test_fcst["yearly"], mode="lines", name="Yearly Seasonality"), row=3, col=1)
    fig9.add_trace(go.Scatter(x=test_fcst["ds"], y=test_fcst["daily"], mode="lines", name="Daily Seasonality"), row=4, col=1)
    # Customize the layout of the figure
    fig9.update_layout(title=f"Prophet Seasonal Components - Water Height Over Time - {TITLE_POSTFIX_NAME}",
                    title_x=0.5,
                        title_font=dict(size=24, color="black", family="Arial", weight="bold"),
                        xaxis_title="Datetime", 
                        yaxis_title="Component Value",
                        legend_title="Legend",
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)", bordercolor="black", borderwidth=1),
                        height=900, width=1200)
    return fig9



# *============================================================
# *Implementation of LSTM (START)
# *============================================================

# LSTM (Long Short-Term Memory)
# it's a type of recurrent neural network (RNN) that is well-suited for time series forecasting tasks. LSTMs are designed to capture long-term dependencies in sequential data, making them effective for modeling complex patterns in time series data.

# # specify to use GPU or CPU device
# import tensorflow as tf
# # Check if GPU is available and set the device accordingly
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     device_name = '/device:CPU:0'
# print(f"Using device: {device_name}")

# # set CPU or GPU device for TensorFlow operations
# tf.config.set_visible_devices([], 'GPU') if device_name == '/device:CPU:0' else tf.config.set_visible_devices([], 'CPU')

# ============================================================
# FONCTION METRIQUE SMAPE
# ============================================================

# Définition d'une fonction utilitaire pour calculer la sMAPE
def smape(y_true, y_pred):
    # Conversion en tableau NumPy
    y_true = np.array(y_true)
    # Conversion en tableau NumPy
    y_pred = np.array(y_pred)
    # Calcul du dénominateur
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Eviter les divisions par zéro
    denominator = np.where(denominator == 0, 1e-8, denominator)
    # Retour de la sMAPE en pourcentage
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


# ============================================================
# FONCTION PRINCIPALE LSTM_prediction
# ============================================================

def LSTM_prediction(data_df,
                    LSTM_parameters = {"date_col": "ds",
                                        "target_col": "y",
                                        "look_back": 30,
                                        "test_size": 0.20,
                                        "epochs": 60,
                                        "batch_size": 32,
                                        "lstm_units_1": 64,
                                        "lstm_units_2": 32,
                                        "dropout_rate": 0.20,
                                        "learning_patience": 5,
                                        "early_stopping_patience": 10,
                                        "resample_freq": "D",
                                        "interpolate_method": "time",
                                        "use_exogenous_features": True,
                                        "verbose": 1,
                                        "random_seed": 42
                                    }) -> dict:
    """
    Fonction générique pour entraîner et évaluer un modèle LSTM sur série temporelle.

    Paramètres
    ----------
    data_df : pd.DataFrame
        DataFrame contenant la série temporelle.
    date_col : str
        Nom de la colonne date.
    target_col : str
        Nom de la colonne cible.
    look_back : int
        Longueur de la fenêtre temporelle (nombre de pas passés utilisés pour prédire le suivant).
    test_size : float
        Taille du jeu de test sous forme de proportion.
    epochs : int
        Nombre maximum d'époques d'entraînement.
    batch_size : int
        Taille des batchs.
    lstm_units_1 : int
        Nombre de neurones de la première couche LSTM.
    lstm_units_2 : int
        Nombre de neurones de la seconde couche LSTM.
    dropout_rate : float
        Taux de dropout.
    learning_patience : int
        Patience pour ReduceLROnPlateau.
    early_stopping_patience : int
        Patience pour EarlyStopping.
    resample_freq : str
        Fréquence de rééchantillonnage (ex: 'D', 'H', 'M').
    interpolate_method : str
        Méthode d'interpolation.
    use_exogenous_features : bool
        Si True, utilise toutes les colonnes numériques disponibles en plus de la target.
    verbose : int
        Niveau de verbosité Keras.
    random_seed : int
        Graine aléatoire.

    Retour
    ------
    results : dict
        Dictionnaire contenant :
        - model : modèle Keras entraîné
        - metrics_df : DataFrame des métriques
        - forecast_df : DataFrame des prédictions alignées en dates
        - history_df : historique des losses
        - train_df : DataFrame train
        - test_df : DataFrame test
        - eda_summary : résumé EDA
        - figures : dictionnaire de figures matplotlib
        - feature_columns : colonnes utilisées comme features
        - scalers : dictionnaire des scalers
    """

    date_col = LSTM_parameters["date_col"]
    target_col = LSTM_parameters["target_col"]
    look_back = LSTM_parameters["look_back"]
    test_size = LSTM_parameters["test_size"]
    epochs = LSTM_parameters["epochs"]
    batch_size = LSTM_parameters["batch_size"]
    lstm_units_1 = LSTM_parameters["lstm_units_1"]
    lstm_units_2 = LSTM_parameters["lstm_units_2"]
    dropout_rate = LSTM_parameters["dropout_rate"]
    learning_patience = LSTM_parameters["learning_patience"]
    early_stopping_patience = LSTM_parameters["early_stopping_patience"]
    resample_freq = LSTM_parameters["resample_freq"]
    interpolate_method = LSTM_parameters["interpolate_method"]
    use_exogenous_features = LSTM_parameters["use_exogenous_features"]
    verbose = LSTM_parameters["verbose"]
    random_seed = LSTM_parameters["random_seed"]

    # ============================================================
    # 1) SECURITE ET REPRODUCTIBILITE
    # ============================================================

    # Fixer la graine NumPy pour reproductibilité
    np.random.seed(random_seed)

    # Fixer la graine TensorFlow
    tf.random.set_seed(random_seed)

    # ============================================================
    # 2) COPIE ET VALIDATION DES DONNEES
    # ============================================================

    # Créer une copie du DataFrame pour éviter toute modification externe
    df = data_df.copy()
    # print(f"Columns in the original DataFrame: {df.columns.tolist()}")  # Debug: afficher les colonnes disponibles

    # Vérifier que la colonne date existe
    if date_col not in df.columns:
        raise ValueError(f"La colonne date '{date_col}' est absente du DataFrame.")

    # Vérifier que la colonne target existe
    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' est absente du DataFrame.")

    # ============================================================
    # 3) PREPARATION DES DONNEES
    # ============================================================

    # Convertir la colonne date en datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Convertir la cible en numérique
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Supprimer les lignes avec date ou cible manquantes
    df = df.dropna(subset=[date_col, target_col])

    # Trier par ordre chronologique
    df = df.sort_values(date_col).reset_index(drop=True)

    # Définir la colonne date comme index
    df = df.set_index(date_col)

    # Si plusieurs observations à la même date ou si la fréquence est irrégulière,
    # on rééchantillonne pour avoir une série régulière
    df = df.resample(resample_freq).mean()

    # Interpoler les valeurs manquantes temporelles
    df = df.interpolate(method=interpolate_method)

    # Effectuer un forward fill si besoin résiduel
    df = df.ffill()

    # Effectuer un backward fill pour les trous initiaux éventuels
    df = df.bfill()

    # ============================================================
    # 4) SELECTION DES FEATURES
    # ============================================================

    # Si on veut utiliser des variables explicatives additionnelles
    if use_exogenous_features:
        # Garder uniquement les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # S'assurer que la target est incluse
        if target_col not in numeric_cols:
            numeric_cols.append(target_col)
        # Définir les colonnes finales utilisées
        feature_columns = numeric_cols
    else:
        # Utiliser uniquement la cible
        feature_columns = [target_col]

    # Créer un DataFrame restreint aux features utiles
    df_model = df[feature_columns].copy()

    # ============================================================
    # 5) EDA INTEGREE A LA FONCTION
    # ============================================================

    # Créer un dictionnaire qui contiendra les figures
    figures = {}

    # Créer un dictionnaire résumé EDA
    eda_summary = {}

    # Résumé dimensions
    eda_summary["n_rows"] = int(df_model.shape[0])

    # Résumé nombre de colonnes
    eda_summary["n_features"] = int(df_model.shape[1])

    # Date minimum
    eda_summary["start_date"] = str(df_model.index.min())

    # Date maximum
    eda_summary["end_date"] = str(df_model.index.max())

    # Nombre de NA résiduelles
    eda_summary["missing_values_total"] = int(df_model.isna().sum().sum())

    # Statistiques descriptives
    eda_summary["describe"] = df_model.describe().to_dict()

    # ---------- Figure 1 : série temporelle brute ----------
    # Créer la figure
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    # Tracer la cible
    ax1.plot(df_model.index, df_model[target_col], color="royalblue", linewidth=1.5)
    # Ajouter un titre
    ax1.set_title("EDA - Série temporelle de la variable cible")
    # Ajouter le label X
    ax1.set_xlabel("Date")
    # Ajouter le label Y
    ax1.set_ylabel(target_col)
    # Ajuster le layout
    fig1.tight_layout()
    # Sauvegarder la figure en mémoire
    figures["time_series"] = fig1

    # ---------- Figure 2 : distribution cible ----------
    # Créer la figure
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    # Tracer un histogramme
    sns.histplot(df_model[target_col], kde=True, ax=ax2, color="darkorange")
    # Ajouter un titre
    ax2.set_title("EDA - Distribution de la cible")
    # Ajuster le layout
    fig2.tight_layout()
    # Sauvegarder la figure
    figures["target_distribution"] = fig2

    # ---------- Figure 3 : rolling mean / rolling std ----------
    # Créer une série lissée
    rolling_mean = df_model[target_col].rolling(window=min(look_back, 30)).mean()
    # Créer une série d'écart-type roulant
    rolling_std = df_model[target_col].rolling(window=min(look_back, 30)).std()
    # Créer la figure
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    # Tracer la série brute
    ax3.plot(df_model.index, df_model[target_col], label="Original", alpha=0.5)
    # Tracer la moyenne glissante
    ax3.plot(df_model.index, rolling_mean, label="Rolling Mean", linewidth=2)
    # Tracer l'écart-type glissant
    ax3.plot(df_model.index, rolling_std, label="Rolling Std", linewidth=2)
    # Ajouter un titre
    ax3.set_title("EDA - Rolling Mean & Rolling Std")
    # Ajouter la légende
    ax3.legend()
    # Ajuster le layout
    fig3.tight_layout()
    # Sauvegarder la figure
    figures["rolling_stats"] = fig3

    # ============================================================
    # 6) SPLIT TEMPOREL TRAIN / TEST
    # ============================================================

    # Calculer l'index de split chronologique
    split_idx = int(len(df_model) * (1 - test_size))

    # Sous-ensemble train
    train_df = df_model.iloc[:split_idx].copy()

    # Sous-ensemble test
    test_df = df_model.iloc[split_idx:].copy()

    # Vérifier qu'il y a assez d'observations pour le look_back
    if len(train_df) <= look_back or len(test_df) <= 1:
        raise ValueError("Pas assez de données pour appliquer la fenêtre look_back choisie.")

    # ============================================================
    # 7) SCALING
    # ============================================================

    # Créer un scaler pour les features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit du scaler uniquement sur le train pour éviter les fuites de données
    train_scaled = feature_scaler.fit_transform(train_df)

    # Transformation du test avec le scaler fit sur train
    test_scaled = feature_scaler.transform(test_df)

    # Récupérer l'index de la colonne cible dans les features
    target_idx = feature_columns.index(target_col)

    # ============================================================
    # 8) CREATION DES SEQUENCES LSTM
    # ============================================================

    # Définir une fonction interne pour créer des séquences
    def create_sequences(data_array, look_back_steps, target_index):
        # Initialiser liste X
        X = []
        # Initialiser liste y
        y = []
        # Boucler sur les lignes après la fenêtre look_back
        for i in range(look_back_steps, len(data_array)):
            # Ajouter la fenêtre passée à X
            X.append(data_array[i - look_back_steps:i, :])
            # Ajouter la valeur cible à prédire à y
            y.append(data_array[i, target_index])
        # Retourner sous forme de tableaux NumPy
        return np.array(X), np.array(y)

    # Créer les séquences train
    X_train, y_train = create_sequences(train_scaled, look_back, target_idx)

    # Pour le test, on concatène la fin du train et le test pour garder le contexte
    full_test_input = np.vstack([train_scaled[-look_back:], test_scaled])

    # Créer les séquences test
    X_test, y_test = create_sequences(full_test_input, look_back, target_idx)

    # Construire les dates associées aux prédictions test
    test_prediction_dates = test_df.index

    # ============================================================
    # 9) CONSTRUCTION DU MODELE LSTM
    # ============================================================

    # Créer un modèle séquentiel Keras
    model = Sequential()

    # Ajouter la couche d'entrée explicite
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    # Ajouter une première couche LSTM qui renvoie la séquence complète
    model.add(LSTM(units=lstm_units_1, return_sequences=True))

    # Ajouter du dropout pour réduire le sur-apprentissage
    model.add(Dropout(dropout_rate))

    # Ajouter une seconde couche LSTM qui renvoie uniquement le dernier état caché
    model.add(LSTM(units=lstm_units_2, return_sequences=False))

    # Ajouter un second dropout
    model.add(Dropout(dropout_rate))

    # Ajouter une couche dense intermédiaire
    model.add(Dense(16, activation="relu"))

    # Ajouter la couche de sortie pour prédire une seule valeur
    model.add(Dense(1))

    # Compiler le modèle avec Adam et la loss MSE
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # ============================================================
    # 10) CALLBACKS
    # ============================================================

    # Callback EarlyStopping pour stopper si la validation n'améliore plus
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True
    )

    # Callback ReduceLROnPlateau pour réduire le learning rate si stagnation
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=learning_patience,
        min_lr=1e-6
    )

    # ============================================================
    # 11) ENTRAINEMENT
    # ============================================================

    # Entraîner le modèle
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.10,
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose,
        shuffle=False
    )

    # Convertir l'historique en DataFrame
    history_df = pd.DataFrame(history.history)

    # ============================================================
    # 12) PREDICTIONS
    # ============================================================

    # Faire les prédictions sur les séquences test
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Pour inverser l'échelle, on doit reconstruire des matrices complètes de features
    # avec la target prédite à la bonne position
    y_pred_full = np.zeros((len(y_pred_scaled), len(feature_columns)))

    # Placer la target prédite dans la bonne colonne
    y_pred_full[:, target_idx] = y_pred_scaled.flatten()

    # Inverser la normalisation
    y_pred_inversed = feature_scaler.inverse_transform(y_pred_full)[:, target_idx]

    # Faire la même chose pour y_test réel
    y_test_full = np.zeros((len(y_test), len(feature_columns)))

    # Injecter la vraie target scalée
    y_test_full[:, target_idx] = y_test

    # Inverser la normalisation
    y_test_inversed = feature_scaler.inverse_transform(y_test_full)[:, target_idx]

    # ============================================================
    # 13) DATAFRAME DE PREVISION
    # ============================================================

    # Construire le DataFrame final de comparaison
    forecast_df = pd.DataFrame({
        "ds": test_prediction_dates,
        "y_true": y_test_inversed,
        "y_pred_lstm": y_pred_inversed
    })

    # Calculer le résidu
    forecast_df["residual"] = forecast_df["y_true"] - forecast_df["y_pred_lstm"]

    # ============================================================
    # 14) METRIQUES
    # ============================================================

    # Calcul du RMSE
    rmse_value = np.sqrt(mean_squared_error(forecast_df["y_true"], forecast_df["y_pred_lstm"]))

    # Calcul du MAE
    mae_value = mean_absolute_error(forecast_df["y_true"], forecast_df["y_pred_lstm"])

    # Calcul du R2
    r2_value = r2_score(forecast_df["y_true"], forecast_df["y_pred_lstm"])

    # Calcul du MAPE
    mape_value = np.mean(
        np.abs(
            (forecast_df["y_true"] - forecast_df["y_pred_lstm"]) /
            np.clip(np.abs(forecast_df["y_true"]), 1e-8, None)
        )
    ) * 100

    # Calcul du sMAPE
    smape_value = smape(forecast_df["y_true"], forecast_df["y_pred_lstm"])

    # Calcul du biais moyen
    bias_value = np.mean(forecast_df["y_pred_lstm"] - forecast_df["y_true"])

    # Créer un DataFrame de métriques
    metrics_df = pd.DataFrame({
        "model": ["LSTM"],
        "RMSE": [rmse_value],
        "MAE": [mae_value],
        "R2": [r2_value],
        "MAPE_%": [mape_value],
        "sMAPE_%": [smape_value],
        "Bias": [bias_value]
    })

    # ============================================================
    # 15) FIGURES DE RESULTATS
    # ============================================================

    # ---------- Figure 4 : train / test / prédictions ----------
    # Créer la figure
    fig4, ax4 = plt.subplots(figsize=(15, 6))
    # Tracer la cible train
    ax4.plot(train_df.index, train_df[target_col], label="Train", alpha=0.6)
    # Tracer la cible test
    ax4.plot(forecast_df["ds"], forecast_df["y_true"], label="Test réel", linewidth=2)
    # Tracer la prédiction LSTM
    ax4.plot(forecast_df["ds"], forecast_df["y_pred_lstm"], label="Prédiction LSTM", linewidth=2)
    # Ajouter le titre
    ax4.set_title("Prévision LSTM vs valeurs réelles")
    # Ajouter la légende
    ax4.legend()
    # Ajuster le layout
    fig4.tight_layout()
    # Sauvegarder la figure
    figures["forecast_comparison"] = fig4

    # ---------- Figure 5 : courbes d'entraînement ----------
    # Créer la figure
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    # Tracer la loss train
    ax5.plot(history_df["loss"], label="Train Loss")
    # Tracer la loss validation
    ax5.plot(history_df["val_loss"], label="Validation Loss")
    # Ajouter le titre
    ax5.set_title("Historique d'entraînement LSTM")
    # Ajouter la légende
    ax5.legend()
    # Ajuster le layout
    fig5.tight_layout()
    # Sauvegarder la figure
    figures["training_history"] = fig5

    # ============================================================
    # 16) PREPARATION DU RESULTAT FINAL
    # ============================================================

    # Construire le dictionnaire de sortie
    results = {
        "model": model,
        "metrics_df": metrics_df,
        "forecast_df": forecast_df,
        "history_df": history_df,
        "train_df": train_df.reset_index(),
        "test_df": test_df.reset_index(),
        "eda_summary": eda_summary,
        "figures": figures,
        "feature_columns": feature_columns,
        "scalers": {
            "feature_scaler": feature_scaler
        }
    }

    # Retourner tous les résultats
    return results


def figure_LSTM(lstm_results, train_fbp, test_fbp, prophet_forecast, STATION_MEASURE, TITLE_POSTFIX_NAME):
    if prophet_forecast is not None:
        # Fusionner les DataFrames de prédictions LSTM et Prophet sur la colonne 'ds' pour comparer les prédictions
        comparison_df = lstm_results["forecast_df"].merge(prophet_forecast[['ds', 'yhat']], on='ds', how='inner')
        # Renommer la colonne 'yhat' pour indiquer qu'il s'agit des prédictions de Prophet
        comparison_df.rename(columns={"yhat": "y_pred_prophet"}, inplace=True)
    else:
        comparison_df = lstm_results["forecast_df"].copy()
    # =============================================== fig LSTM
    # Plot the prophet_forecast with confidence intervals and actual history using plotly with interactive plot and range slider options
    # We visualize the prophet_forecast, confidence intervals, and actual history.
    # Use plotly only to visualize the prophet_forecast, confidence intervals, and actual history with interactive plot and range slider options
    fig_lstm = go.Figure()
    # Add the actual training data
    fig_lstm.add_trace(go.Scatter(x=train_fbp['ds'], y=train_fbp['y'], mode='lines+markers', name='Actual (Training)', line=dict(color='blue')))
    # Add the actual test data
    fig_lstm.add_trace(go.Scatter(x=test_fbp['ds'], y=test_fbp['y'], mode='lines+markers', name='Actual (Test)', line=dict(color='green')))
    # Add the prophet_forecast line
    if prophet_forecast is not None:
        fig_lstm.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='Prophet Forecast', line=dict(color='red')))
        # Add confidence interval upper bound (invisible line to define fill boundary)
        fig_lstm.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'], fill=None, mode='lines', 
                                line_color='rgba(0,0,0,0)', showlegend=False))
        # Add confidence interval lower bound with fill to upper bound
        fig_lstm.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'], fill='tonexty', mode='lines', 
                                line_color='rgba(0,0,0,0)', name='Prophet Confidence Interval', 
                                fillcolor='rgba(255,0,0,0.2)'))

    # LSTM forecast line
    fig_lstm.add_trace(go.Scatter(x=comparison_df['ds'], y=comparison_df['y_pred_lstm'], mode='lines', name='LSTM Forecast', line=dict(color="cyan")))

    # add horizontal line for flood threshold (example: 2.86) in April 1983
    fig_lstm.add_hline(y=2.86*1000, line_dash="dash", line_color="red", annotation_text=f"Flood Threshold ({2.86*1000}) in April 1983", annotation_position="top left")
    # add horizontal line for flood threshold (example: 2.77) in February 1990
    fig_lstm.add_hline(y=2.77*1000, line_dash="dash", line_color="cyan", annotation_text=f"Flood Threshold ({2.77*1000}) in February 1990", annotation_position="bottom left")
    # add horizontal line for flood threshold (example: 2.58) in January 2018
    fig_lstm.add_hline(y=2.58*1000, line_dash="dash", line_color="green", annotation_text=f"Flood Threshold ({2.58*1000}) in January 2018", annotation_position="bottom left")

    # Update layout with range selector buttons and range slider
    fig_lstm.update_layout(
        title=f"LSTM (30 days) - Prophet Forecast | LSTM - Water {STATION_MEASURE} Over Time - {TITLE_POSTFIX_NAME}",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title=f"Water {STATION_MEASURE}",
        legend_title="Legend",
        width=1280,
        height=600,
        xaxis=dict(
            # Define the range selector with buttons
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            # Add a range slider to the x-axis
            rangeslider=dict(visible=True),
            # Set the x-axis type to date
            type="date"
        )
    )
    return fig_lstm


# *============================================================
# *Implementation of LSTM (END)
# *============================================================


# run script with arguments example: python test.py --test-size 0.25 --random_state 123
# for comparison with Prophet set <prophet_forecast> which is an optional argument that can be passed to the main_train_lstm function if you want to include the Prophet forecast in the LSTM forecast plot. 
# If you have already trained a Prophet model and generated a forecast, you can pass that forecast as a DataFrame to this function to visualize it alongside the LSTM predictions. If you do not have a Prophet forecast or do not want to include it in the plot, you can simply call main_train_lstm() without any arguments, 
# and it will proceed with training the LSTM model and plotting the results without the Prophet forecast.
def main_train_lstm(prophet_forecast:Prophet = None) -> dict:
    args = parse_args()
    NUM_CPUS = args.ncpus #os.cpu_count()
    RANDOM_STATE = args.random_state
    seasonality = args.seasonality
    changepoint_prior_scale = args.changepoint_prior_scale
    seasonality_prior_scale = args.seasonality_prior_scale
    
    # Load Data
    site_name = args.site_name
    code_site = args.code_site
    code_station = args.code_station
    site_measure = args.site_measure
    start_date = args.start_date
    end_date = args.end_date
    
    default_code_site = SITES[site_name]["code"]  # Read default site code from configuration.
    if code_site != default_code_site:
        code_site = default_code_site  # Override with default site code.
    station_codes = [item["code"] for item in SITES[site_name]["stations"]]  # Build station options list for selected site.
    default_station = station_codes[0]
    if code_station not in station_codes:
        code_station = default_station  # Pick first station as default.

    
    # Set the split date for train/test split 
    # (example: "2024-01-01" to use all data up to the end of 2023 for training and the rest for testing) 
    SPLIT_DATE = "2024-01-01"
    
    TITLE_POSTFIX_NAME = f"{site_name} - {code_site} - {code_station} : {site_measure}"
    FILE_POSTFIX_NAME = TITLE_POSTFIX_NAME.replace(" ", "").replace(":", "_").replace("-", "_").replace("(", "").replace(")", "")
    
    
    # # # Set tracking URI to your Hugging Face application (optional, defaults to local file-based storage)
    # Tell MLflow to use the Neon database to log parameters, metrics, and run names
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Keep the model registry endpoint explicit so both run tracking and model registration hit the same intended service.
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    
    # Use one stable experiment name from env instead of timestamped experiment names to avoid creating multiple experiments in the Neon DB and instead log all runs under a single experiment for better organization and comparison of results.
    MLFLOW_EXPERIMENT_NAME = "final_project_forecasting_hubeau/lstm"
    REGISTERED_MODEL_NAME = "flood_forecast_model_lstm"
    # Alias for the registered model version, can be used to point to the latest version or a specific version for easier reference in deployment and inference.
    # Use a stable (no timestamped) alias for deployment/inference, or a timestamped alias for tracking different versions over time. Here we use a timestamped alias to keep track of different model versions based on the training time.
    # MODEL_ALIAS_NAME = f"challenge_{Current_time}" 
    MODEL_ALIAS_NAME = "challenge"
    
    print(f"MLFLOW_EXPERIMENT_NAME: {MLFLOW_EXPERIMENT_NAME}")
    print(f"REGISTERED_MODEL_NAME: {REGISTERED_MODEL_NAME}")
    print(f"MODEL_ALIAS_NAME: {MODEL_ALIAS_NAME}")

    # Attempt to retrieve the experiment by its name to see if it already exists in the Neon DB
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    # Check if the experiment does not exist yet
    if experiment is None:
        # Only force an artifact location when one is explicitly configured; for an MLflow server, the server artifact root is usually the correct source of truth.
        if MLFLOW_ARTIFACT_URI:
            mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_URI)
        else:
            mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
    
    # Set the newly created or existing experiment as the active one for this script
    # set mlflow experiment name (optional, will create if it doesn"t exist)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Initialize the MLflow client to interact with the tracking server (Neon DB) 
    # tracking_uri: the URI of the MLflow tracking server (Neon DB URL) - Address of local or remote tracking server. If not provided, defaults to the service set by mlflow.tracking.set_tracking_uri. See Where Runs Get Recorded for more info.
    # registry_uri: the URI of the MLflow registry server (Neon DB URL) -     Address of local or remote model registry server. If not provided, defaults to the service set by mlflow.tracking.set_registry_uri. If no such service was set, defaults to the tracking uri of the client.
    # workspace_store_uri: the URI of the MLflow workspace store server (Neon DB URL) - Address of the workspace provider backend. Defaults to the tracking URI when unspecified, but can be pointed at a dedicated workspace store.
    # Do not pass workspace_store_uri unless you specifically need it; it is an advanced option that is not commonly used and can lead to confusion if not set up correctly. In most cases, you can simply use MlflowClient() without any arguments, and it will use the tracking URI for all operations, including model registry interactions.
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_REGISTRY_URI)
    # client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, 
    #                     registry_uri=MLFLOW_TRACKING_URI,
    #                     #workspace_store_uri=BACKEND_STORE_URI
    #                 )
    
    # =======================================================================
    # I) Exploratory Data Analysis (EDA) + Time Series Visualization
    # =======================================================================
    print(f"Loading data for site: {site_name}, code_site: {code_site}, code_station: {code_station}, measure: {site_measure}, from {start_date} to {end_date}...")
    # if site_name:
    #     s3_URI = f"s3://jedha-dz-dsft41/Final_Project_Forecasting/Dataset/hubeau_api/VARS/cleaned/hubeau_obs_elab_{site_name}_{code_site}_{code_station}_{site_measure}_{start_date}_{end_date}.csv"
    #     data_df = pd.read_csv(s3_URI)
    # else:
    data_df = fetch_obs_elab(code_station=code_station, start_date=str(start_date), end_date=str(end_date), measure=site_measure)  # Fetch raw API data.
    
    # I.1. Select the relevant columns and rename them to "ds" and "y"
    # I.2. Ensure the "ds" column is a proper datetime type.
    # Note: Prophet often prefers timezone-naive datetimes, so we remove the timezone if it exists.
    clean_data_df = prepare_series(data_df)  # Clean and standardize raw series.
    
    print(clean_data_df.head())
    print(clean_data_df.tail())

    # =======================================================================
    # II) Train/Test Split of the data for time series forecasting with Prophet
    # =======================================================================
    train_ratio_size = 0.8  # Use 80% of the data for training and 20% for testing.
    train_fbp, test_fbp, SPLIT_DATE = split_train_test(clean_data_df, train_ratio=train_ratio_size, split_by_ratio=True)  # Split data into train and test sets.
    
    # save the train and test sets as csv files
    train_fbp.to_csv(local_artifact_dir / f"train_LSTM_{FILE_POSTFIX_NAME}.csv", index=False)
    test_fbp.to_csv(local_artifact_dir / f"test_LSTM_{FILE_POSTFIX_NAME}.csv", index=False)

    print("\n" + "="*100)
    print("Starting LSTMn evaluation in nested run...")
    print("="*100 + "\n")    
    # ===========================================================================================
    # Start a parent MLflow run for the entire experiment to organize both baseline and CV models
    # as nested runs for clearer comparison and organization in the MLflow UI.
    # ===========================================================================================
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=f"lstm_station_{code_station}") as lstm_run:
        # Log shared experiment-level parameters
        mlflow.log_param("site_name", site_name)
        mlflow.log_param("data_split_date", SPLIT_DATE)
        
        fig1 = plot_time_series_with_plotly(clean_data_df, site_measure, f"LSTM_{TITLE_POSTFIX_NAME}")
        mlflow.log_figure(fig1, f"Plot/01_time_series_LSTM_{FILE_POSTFIX_NAME}.html")
        
        fig2 = plot_train_test_split(train_fbp, test_fbp, SPLIT_DATE, site_measure, f"LSTM_{TITLE_POSTFIX_NAME}")
        mlflow.log_figure(fig2, f"Plot/02_train_test_split_LSTM_{FILE_POSTFIX_NAME}.html")
        
        # Log the train/test split visualization and data files as parent-level artifacts
        mlflow.log_artifacts(str(local_artifact_dir))
        
        start_time = time.time()
        print(f"{start_time} -> Training baseline model...")
        # =======================================================================
        # III) LSTM Model Evaluation
        # =======================================================================
        
        # LSTM parameters:
        
        LSTM_parameters_dict = {"date_col": "ds",
                                "target_col": "y",
                                "look_back": 30,
                                "test_size": 1 - train_ratio_size,
                                "epochs": 60,
                                "batch_size": 32,
                                "lstm_units_1": 64,
                                "lstm_units_2": 32,
                                "dropout_rate": 0.20,
                                "learning_patience": 5,
                                "early_stopping_patience": 10,
                                "resample_freq": "D",
                                "interpolate_method": "time",
                                "use_exogenous_features": True,
                                "verbose": 1,
                                "random_seed": RANDOM_STATE
                            }
        
        print(f"LSTM parameters: {LSTM_parameters_dict}")
        print(clean_data_df.head())
        
        # Route to LSTM pipeline.
        # prediction avec la fonction LSTM_prediction
        lstm_results = LSTM_prediction(clean_data_df,
                                    LSTM_parameters = LSTM_parameters_dict,
                                )

        # Display the summary of the LSTM model architecture
        lstm_results["model"].summary()

        # Afficher les métriques de LSTM: model       RMSE        MAE       R2    MAPE_%   sMAPE_%      Bias
        print(lstm_results["metrics_df"])

        # Voir les prédictions
        print(lstm_results["forecast_df"].head())

        # PLot LSTM prediction 
        forecast = None  # Placeholder for the Prophet forecast DataFrame, which should be defined earlier in the code where Prophet is run.
        fig_lstm = figure_LSTM(lstm_results, train_fbp, test_fbp, forecast, local_artifact_dir, f"LSTM_{FILE_POSTFIX_NAME}_{Current_time}")
        mlflow.log_figure(fig_lstm, f"Plot/03_forecast_comparison_LSTM_{FILE_POSTFIX_NAME}.html")
        
        # Log LSTM validation metrics
        lstm_metrics = lstm_results["metrics_df"]
        if lstm_metrics is None:
            raise ValueError("performance_metrics returned None; LSTM validation metrics could not be computed.")
        
        # Calculate mean metrics across all rows (if multiple) to log a single value per metric in MLflow        
        lstm_mean_metrics = lstm_metrics[["RMSE", "MAE", "R2", "MAPE_%", "sMAPE_%"]].mean().to_dict()
        
        # Sanitize metric names (MLflow doesn't allow '%' in metric names)
        # Replace '%' with 'pct' to maintain clarity while complying with MLflow restrictions
        sanitized_metrics = {}
        for metric_name, metric_value in lstm_mean_metrics.items():
            # Replace '%' with 'pct' and create valid MLflow metric names
            clean_name = f"lstm_{metric_name.replace('%', 'pct')}"
            sanitized_metrics[clean_name] = float(metric_value)
        
        # Log metrics individually for better error handling
        try:
            mlflow.log_metrics(sanitized_metrics)
        except Exception as e:
            print(f"[WARNING] Failed to log metrics to MLflow: {e}")
            # Log metrics individually as fallback
            for metric_name, metric_value in sanitized_metrics.items():
                try:
                    mlflow.log_metric(metric_name, metric_value)
                except Exception as e_individual:
                    print(f"[WARNING] Failed to log individual metric {metric_name}: {e_individual}")
        
        # Log LSTM model parameters
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("run_type", "lstm_validation")
        for key, value in LSTM_parameters_dict.items():
            mlflow.log_param(key, value)
        
        # Log the LSTM model to MLflow
        model_info_lstm =  mlflow.tensorflow.log_model(lstm_results["model"], 
                                                name=f"lstm_model_{code_station}",
                                                registered_model_name=f"{REGISTERED_MODEL_NAME}",
                                                input_example=train_fbp.drop(columns=["y"]).iloc[:5],
                                                # signature=infer_signature(train_fbp.drop(columns=["y"]).iloc[:5], 
                                                #                         lstm_results["forecast_df"]["y_pred_lstm"].iloc[:5].to_numpy().reshape(-1, 1))
                                            )
        
        # Tag for tracking
        mlflow.set_tags({
            "model_type": "lstm",
            "project": "flood_forecast",
            "station_id": code_station,
            "site_name": site_name,
            "site_measure": site_measure,
            "code_site": code_site,
            "dataset": "hubeau_api", 
            "framework": "tensorflow",
            "training_time_seconds": time.time() - start_time,
            "training_time_minutes": (time.time() - start_time) / 60,
            "training_time_hours": (time.time() - start_time) / 3600
        })
        
        # set registered model description with the main parameters and metrics of the model
        registered_lstm = mlflow.register_model(model_uri=model_info_lstm.model_uri,
                                                name=f"{REGISTERED_MODEL_NAME}", 
                                            )
        
        lstm_model_version = int(registered_lstm.version)
        print(f"\n[INFO] Model logged as version {lstm_model_version}")

        # Set alias for easy model lookup & deployment
        client.set_registered_model_alias(name=f"{REGISTERED_MODEL_NAME}",
                                        alias=MODEL_ALIAS_NAME,
                                        version=str(lstm_model_version)
                                    )
        print(f"[INFO] Alias '{MODEL_ALIAS_NAME}' now points to version {lstm_model_version}")
        
        # Log LSTM results and metrics as CSV files
        lstm_metrics_csv_path = local_artifact_dir / f"lstm_metrics_{FILE_POSTFIX_NAME}_{Current_time}.csv"
        lstm_metrics.to_csv(lstm_metrics_csv_path, index=False)
        # Log LSTM results to MLflow
        mlflow.log_artifact(str(lstm_metrics_csv_path))
        
        # Log LSTM model summary
        print(f"[INFO] LSTM Validation Model logged")
        print(f"[INFO] LSTM Run ID: {lstm_run.info.run_id}")
        print(f"Mean LSTM Metrics:")
        for metric_name, metric_value in lstm_mean_metrics.items():
            print(f"  lstm_{metric_name}: {metric_value:.4f}")
        
    # log the run ID of the LSTM training run as a parameter for easier reference and tracking in MLflow
    mlflow.log_param("lstm_run_id", lstm_run.info.run_id)
    
    return lstm_results

if __name__ == "__main__":
    main_train_lstm()