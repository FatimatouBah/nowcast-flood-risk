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

warnings.filterwarnings("ignore")  # Hide non-critical warnings to keep output simple.

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


def prophet_model_cross_validation(model:Prophet) -> pd.DataFrame:
    """Cross-Validation for time series:
    # Run cross-validation on the trained Prophet model to evaluate its performance on different time splits of the data, 
    # which helps assess how well the model generalizes to unseen data and captures temporal patterns. 
    # The parameters for cross-validation are set to evaluate the model"s performance over a horizon of 365 days, 
    # with an initial training period of 730 days and a period of 180 days between each fold.
    """
    cv_results = cross_validation(model,
                                initial="730 days",
                                period="180 days",
                                horizon="365 days",
                            )
    return cv_results


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


def plot_time_series_with_plotly(clean_data_df: pd.DataFrame, STATION_MEASURE: str="", TITLE_POSTFIX_NAME: str="") -> go.Figure:
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


def plot_train_test_split(train_fbp: pd.DataFrame, test_fbp: pd.DataFrame, SPLIT_DATE: str, STATION_MEASURE: str, TITLE_POSTFIX_NAME: str="") -> go.Figure:
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


# *============================================================
# *Implementation of PROPHET (START)
# *============================================================

def run_prophet(train_fbp: pd.DataFrame, test_fbp: pd.DataFrame, seasonality: bool, changepoint_prior_scale: float, seasonality_prior_scale: float) -> Prophet:
    # Set the growth parameter for the Prophet model to "linear" to model a linear trend in the water height data over time.        
    growth = "linear"  # "flat" or "logistic" if you have a cap and floor in your data, but for water height, linear is often more appropriate unless there are known limits to the values.
    
    # Initialize the Prophet model with or without seasonality components (daily, weekly, yearly) based on the seasonality variable
    enabled_seasonality = cast(Any, True) # Enable all seasonality components (daily, weekly, yearly) by default; Prophet will automatically determine which ones to use based on the data and the seasonality_prior_scale parameter. Setting this to True allows Prophet to include any relevant seasonality components without having to specify each one individually.
    if seasonality:
        prophet_model = Prophet(growth=growth,) # = Prophet() if growth is linear
    else:
        if changepoint_prior_scale != 0.05 and seasonality_prior_scale != 10:
            prophet_model = Prophet(growth=growth,
                                    changepoint_prior_scale=changepoint_prior_scale,
                                    seasonality_prior_scale=seasonality_prior_scale,
                                    yearly_seasonality=enabled_seasonality,
                                    weekly_seasonality=enabled_seasonality,
                                    daily_seasonality=enabled_seasonality,
                                )
        else:
            prophet_model = Prophet(growth=growth,
                                    daily_seasonality=enabled_seasonality, 
                                    weekly_seasonality=enabled_seasonality, 
                                    yearly_seasonality=enabled_seasonality)

    # Fit the Prophet model on the training data
    prophet_model.fit(train_fbp)
    return prophet_model

# *============================================================
# *Implementation of PROPHET (END)
# *============================================================

def plot_prophet_components(test_fcst: pd.DataFrame, TITLE_POSTFIX_NAME: str):
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


def plot_forecast_with_plotly(forecast: pd.DataFrame, TITLE_POSTFIX_NAME: str=""):
    # Plot the forecasted results using plotly
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecasted Water Height (yhat)"))
    fig6.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Confidence Interval (yhat_lower)", line=dict(dash="dash", color="red")))
    fig6.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Confidence Interval (yhat_upper)", line=dict(dash="dash", color="green")))
    # Customize the layout of the figure
    fig6.update_layout(title=f"Prophet Forecast - Water Height Over Time - {TITLE_POSTFIX_NAME}",
                    title_x=0.5,
                        title_font=dict(size=24, color="black", family="Arial", weight="bold"),
                        xaxis_title="Datetime", 
                        yaxis_title="Water Height",
                        legend_title="Legend",
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)", bordercolor="black", borderwidth=1),
                        height=900, width=1200)
    return fig6


def plot_forecast_with_history(forecast: pd.DataFrame, train_fbp: pd.DataFrame, test_fbp: pd.DataFrame, TITLE_POSTFIX_NAME: str="") -> go.Figure:
    # Plot the forecast with confidence intervals and actual history using plotly with interactive plot and range slider options
    # We visualize the forecast, confidence intervals, and actual history.
    # Use plotly only to visualize the forecast, confidence intervals, and actual history with interactive plot and range slider options
    fig7 = go.Figure()
    # Add the actual training data
    fig7.add_trace(go.Scatter(x=train_fbp["ds"], y=train_fbp["y"], mode="lines+markers", name="Actual (Training)", line=dict(color="blue")))
    # Add the actual test data
    fig7.add_trace(go.Scatter(x=test_fbp["ds"], y=test_fbp["y"], mode="lines+markers", name="Actual (Test)", line=dict(color="green")))
    # Add the forecast line
    fig7.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast", line=dict(color="red")))
    # Add confidence interval upper bound (invisible line to define fill boundary)
    fig7.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], fill=None, mode="lines", 
                            line_color="rgba(0,0,0,0)", showlegend=False))
    # Add confidence interval lower bound with fill to upper bound
    fig7.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill="tonexty", mode="lines", 
                            line_color="rgba(0,0,0,0)", name="Confidence Interval", 
                            fillcolor="rgba(255,0,0,0.2)"))
    # Update layout with range selector buttons and range slider
    fig7.update_layout(
        title=f"Prophet Forecast - Water Height Over Time - {TITLE_POSTFIX_NAME}",
        xaxis_title="Date",
        yaxis_title="Water Height",
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
    return fig7


def compute_prophet_metrics(test_fbp: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    # We now measure how accurate the forecast is. If there’s no overlap in dates, it warns you instead of crashing.

    #VI.1. Align predictions with actuals
    # Join actual water height with model predictions on matching dates to compare them directly.

    # Join actual closing prices with model predictions on matching dates to compare them directly.
    # We merge the test dataframe (which contains the actual closing prices for the test period) with the forecast dataframe (which contains the predicted values for the same period) on the "ds" column, 
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


def plot_seasonal_components(test_fcst: pd.DataFrame, TITLE_POSTFIX_NAME: str):
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

# run script with arguments example: python test.py --test-size 0.25 --random-state 123
def main_train_prophet() -> tuple[Prophet, pd.DataFrame, dict[str, float]]:
    #
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
    MLFLOW_EXPERIMENT_NAME = "final_project_forecasting_hubeau/prophet"
    REGISTERED_MODEL_NAME = "flood_forecast_model_prophet"
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
    
    # plot and save the figure as an HTML and png file
    fig1 = plot_time_series_with_plotly(clean_data_df, site_measure, TITLE_POSTFIX_NAME)
    
    # =======================================================================
    # II) Train/Test Split of the data for time series forecasting with Prophet
    # =======================================================================

    train_fbp, test_fbp, SPLIT_DATE = split_train_test(clean_data_df, train_ratio=0.8, split_by_ratio=True)  # Split data into train and test sets.

    # train_fbp = clean_data_df.loc[clean_data_df["ds"] <= SPLIT_DATE].copy()
    # test_fbp = clean_data_df.loc[clean_data_df["ds"] > SPLIT_DATE].copy()
    
    # save the train and test sets as csv files
    train_fbp.to_csv(local_artifact_dir / f"train_fbp_{FILE_POSTFIX_NAME}.csv", index=False)
    test_fbp.to_csv(local_artifact_dir / f"test_fbp_{FILE_POSTFIX_NAME}.csv", index=False)
    # print(train_fbp)
    
    fig2 = plot_train_test_split(train_fbp, test_fbp, SPLIT_DATE, site_measure, TITLE_POSTFIX_NAME)
    
    # ===========================================================================================
    # Start a parent MLflow run for the entire experiment to organize both baseline and CV models
    # as nested runs for clearer comparison and organization in the MLflow UI.
    # ===========================================================================================
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=f"prophet_station_{code_station}") as parent_run:
        # Log shared experiment-level parameters
        mlflow.log_param("site_name", site_name)
        mlflow.log_param("data_split_date", SPLIT_DATE)
        
        # Log the train/test split visualization and data files as parent-level artifacts
        mlflow.log_artifacts(str(local_artifact_dir))
        
        mlflow.log_figure(fig1, artifact_file=f"Plot/01_time_series_plot_{FILE_POSTFIX_NAME}.html")    
        mlflow.log_figure(fig2, artifact_file=f"Plot/02_train_test_split_plot_{FILE_POSTFIX_NAME}.html")

        start_time = time.time()
        print(f"{start_time} -> Training baseline model...")
        # =======================================================================
        # III) Initialize and train the Prophet model
        # =======================================================================
                
        prophet_model = run_prophet(train_fbp, test_fbp, seasonality, changepoint_prior_scale, seasonality_prior_scale)
        
        # mlflow.log_param("train_fbp", train_fbp)
        # mlflow.log_param("test_fbp", test_fbp)
        mlflow.log_param("seasonality", seasonality)
        mlflow.log_param("changepoint_prior_scale", changepoint_prior_scale)
        mlflow.log_param("seasonality_prior_scale", seasonality_prior_scale)
        
        # =======================================================================
        # IV) Predict on test set with model
        # =======================================================================

        test_fbp_prophet = test_fbp.reset_index().rename(columns={DATETIME_COLUMN_NAME:"ds"})
        test_fcst = prophet_model.predict(test_fbp_prophet[["ds"]])
        
        # save the forecasted results as a csv file
        test_fcst.to_csv(local_artifact_dir / f"test_forecast_{FILE_POSTFIX_NAME}.csv", index=False)
        
        fig3 = plot_prophet_components(test_fcst, TITLE_POSTFIX_NAME)
        mlflow.log_figure(fig3, artifact_file=f"Plot/03_prophet_components_plot_{FILE_POSTFIX_NAME}.html")

        # =======================================================================
        # V) Create a future dataframe and forecast
        # =======================================================================

        # # Forecasting the next 24 hours (freq="H") -> 1 day only
        # future = prophet_model.make_future_dataframe(periods=24, freq="H")
        # Forecasting the next 30 days (freq="D") -> 1 month only
        # future = prophet_model.make_future_dataframe(periods=30, freq="D") 
        # # Forecasting the next 365 days (freq="D") -> 1 year only
        # future = prophet_model.make_future_dataframe(periods=365, freq="D")
        future = prophet_model.make_future_dataframe(periods=365*2, freq="D")
        # # Forecasting the next 30 days (freq="D") with daily seasonality only
        # future = prophet_model.make_future_dataframe(periods=30, freq="D", include_history=True)
        # # Forecasting the next 1 year (freq="D") with daily seasonality only
        # future = prophet_model.make_future_dataframe(periods=365, freq="D", include_history=True)

        # Generate the forecast for the future dataframe for the next 24 hours or 30 days
        forecast = prophet_model.predict(future)
        
        fig4 = plot_forecast_with_plotly(forecast, TITLE_POSTFIX_NAME)
        fig5 = plot_forecast_with_history(forecast, train_fbp, test_fbp, TITLE_POSTFIX_NAME)
        mlflow.log_figure(fig4, artifact_file=f"Plot/04_prophet_forecast_plot_{FILE_POSTFIX_NAME}.html")
        mlflow.log_figure(fig5, artifact_file=f"Plot/05_prophet_forecast_with_history_plot_{FILE_POSTFIX_NAME}.html")
        
        # =======================================================================
        # VI) Compute the metrics (MAPE, R^2, RMSE, MAE) to evaluate the forecast accuracy
        # =======================================================================
        
        prophet_metrics = compute_prophet_metrics(test_fbp, forecast)
        mape, r2, mse, rmse, mae = prophet_metrics["std_mape"], prophet_metrics["std_r2"], prophet_metrics["std_mse"], prophet_metrics["std_rmse"], prophet_metrics["std_mae"]
    
        print(f"[INFO] Baseline Run ID: {parent_run.info.run_id}")
        print("===="*20)
        print("Baseline metrics computed successfully.")
        print("===="*20)
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}")
        print(f"R^2 score: {r2:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        
        with open(local_artifact_dir / f"train_{FILE_POSTFIX_NAME}_{Current_time}.txt", "w") as f:
            f.write(f"Current date and time: {Current_time}\n")
            f.write(f"{TITLE_POSTFIX_NAME}\n")
            f.write("Model Type: Prophet\n")
            f.write("\nParameters:\n")
            f.write(f"\t-Seasonality: {args.seasonality}\n")
            f.write("\nThis is an artifact file containing the standard Prophet evaluation metrics of the model:\n")
            for key, value in prophet_metrics.items():
                f.write(f"\t-{key}: {value}\n")
            f.write("\nSettings for the deployment &inference:\n")
            f.write(f"MLflow Experiment Name <MLFLOW_EXPERIMENT_NAME>: {MLFLOW_EXPERIMENT_NAME}\n")
            f.write(f"Registered Model Name <REGISTERED_MODEL_NAME>: {REGISTERED_MODEL_NAME}\n")
            f.write(f"Model Alias Name <MODEL_ALIAS_NAME>: {MODEL_ALIAS_NAME}\n")

        # VI. Plot seasonal components
        # It"s helpful to show how trend, daily, and yearly seasonality patterns contribute to the forecast.
        
        fig6 = plot_seasonal_components(test_fcst, TITLE_POSTFIX_NAME)
        mlflow.log_figure(fig6, artifact_file=f"Plot/06_prophet_seasonal_components_plot_{FILE_POSTFIX_NAME}.html")
        
        # =======================================================================
        # VII) Log model parameters, metrics, and the model itself to MLflow
        # =======================================================================

        # Log global model parameters at parent run level
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_param("seasonality", args.seasonality)
        mlflow.log_param("SPLIT_DATE", SPLIT_DATE)
        
        # ===========================================================================================
        # CHILD RUN 1: Baseline Prophet Model Evaluation on test set
        # ===========================================================================================
        print("\n" + "="*100)
        print("Starting Baseline model evaluation in nested run...")
        print("="*100 + "\n")
        
        with mlflow.start_run(run_name="baseline_test_set", nested=True) as baseline_run:
            mlflow.set_tag("comparison_group", "baseline_vs_cv")
            mlflow.set_tag("evaluation_strategy", "baseline_test_set")

            # Store metrics in MLflow for standard (std_) prophet evaluation metrics (MAPE, R^2, RMSE, MAE) to evaluate the forecast accuracy
            # All metrics are prefixed with "std_" to clearly indicate they are from the standard test-set evaluation
            for key, value in prophet_metrics.items():
                mlflow.log_metric(key, float(value))

            # Log normalized metric keys in both child runs so baseline vs CV can be compared directly in the MLflow UI.
            mlflow.log_metrics({
                "eval_mape": float(mape),
                "eval_r2": float(r2),
                "eval_mse": float(mse),
                "eval_rmse": float(rmse),
                "eval_mae": float(mae),
            })
            
            # Log baseline parameters specific to test-set evaluation
            mlflow.log_param("run_type", "baseline_test_set")
            mlflow.log_param("train_test_split_date", str(SPLIT_DATE))
            
            # Log the model to MLflow with signature and input example for the baseline evaluation
            signature = infer_signature(train_fbp.drop(columns=["y"]), 
                                        prophet_model.predict(train_fbp.drop(columns=["y"]))
                                    )
            
            # Log the baseline model to MLflow with the signature and input example for better reproducibility and to enable MLflow's model validation features. This ensures that the model can be properly loaded and used for inference later on, and that the expected input format is clear.
            model_info = mlflow_prophet.log_model(pr_model=prophet_model, 
                                     name=f"prophet_model_{code_station}",
                                     registered_model_name=REGISTERED_MODEL_NAME,
                                     signature=signature, 
                                     input_example=train_fbp.drop(columns=["y"]).iloc[:5])
            
            # Tag for deployment tracking :: Tag for tracking
            mlflow.set_tags({
                "model_type": "prophet",
                "project": "flood_forecast",
                "station_id": code_station,
                "site_name": site_name,
                "site_measure": site_measure,
                "code_site": code_site,
                "dataset": "hubeau_api", 
                "framework": "prophet",
                "forecast_horizon": "365_days",
                "data_frequency": "daily",
                "training_time_seconds": time.time() - start_time,
                "training_time_minutes": (time.time() - start_time) / 60,
                "training_time_hours": (time.time() - start_time) / 3600
            })
            
            # Log baseline artifacts (plots and forecasts) to the baseline run
            # This keeps baseline-specific artifacts organized separately from CV artifacts
            mlflow.log_artifact(str(local_artifact_dir / f"test_forecast_{FILE_POSTFIX_NAME}.csv"), artifact_path="baseline")
            
            # Save serialized baseline model inside the baseline run context so it"s associated with this run
            serialized_model_dir = local_artifact_dir / "serialized_models"
            if not serialized_model_dir.exists():
                serialized_model_dir.mkdir(parents=True)
            prophet_model_pickle_path = serialized_model_dir / f"prophet_model_{code_station}.pkl"
            prophet_model_joblib_path = serialized_model_dir / f"prophet_model_{code_station}.joblib"
            export_model_with_pickle(prophet_model, str(prophet_model_pickle_path))
            export_model_with_joblib(prophet_model, str(prophet_model_joblib_path))
            mlflow.log_artifacts(str(serialized_model_dir), artifact_path="baseline_serialized")
            
            # Save and Log prophet model to MLflow
            from prophet.serialize import model_to_json, model_from_json
            prophet_model_json_path = serialized_model_dir / f"prophet_model_{code_station}.json"
            with open(prophet_model_json_path, "w") as fout:
                fout.write(model_to_json(prophet_model))  # Save model
            mlflow.log_artifact(str(prophet_model_json_path), artifact_path="baseline_serialized")
            
            # Log baseline model summary
            print(f"[INFO] Baseline Model logged for test-set evaluation")
            print(f"[INFO] Baseline Run ID: {baseline_run.info.run_id}")
            print(f"Mean Baseline Metrics:")
            for key, value in prophet_metrics.items():
                print(f"  {key}: {value:.4f}")

        mlflow.log_param("baseline_child_run_id", baseline_run.info.run_id)
        
        # ============================================================================================
        # VIII) Model Registration and Tagging (at parent run level, after both nested runs)
        # ============================================================================================
        
        # ===========================================================================================
        # CHILD RUN 2: Cross-Validation Model Evaluation
        # ===========================================================================================
        print("\n" + "="*100)
        print("Starting Cross-Validation evaluation in nested run...")
        print("="*100 + "\n")
        
        with mlflow.start_run(run_name="cross_validation", nested=True) as cv_run:
            mlflow.set_tag("comparison_group", "baseline_vs_cv")
            mlflow.set_tag("evaluation_strategy", "cross_validation")

            # Run cross-validation on the trained Prophet model to evaluate performance on different time splits
            cv_results = prophet_model_cross_validation(prophet_model)
            
            # Log cross-validation metrics
            metrics = performance_metrics(cv_results)
            if metrics is None:
                raise ValueError("performance_metrics returned None; cross-validation metrics could not be computed.")
            
            cv_mean_metrics = metrics[["mse", "rmse", "mae", "mape"]].mean().to_dict()
            mlflow.log_metrics({f"cv_{metric_name}": float(metric_value) for metric_name, metric_value in cv_mean_metrics.items()})
            mlflow.log_metrics({
                "eval_mse": float(cv_mean_metrics["mse"]),
                "eval_rmse": float(cv_mean_metrics["rmse"]),
                "eval_mae": float(cv_mean_metrics["mae"]),
                "eval_mape": float(cv_mean_metrics["mape"]),
            })
            
            # Log CV model parameters
            mlflow.log_param("model_type", "Prophet")
            mlflow.log_param("run_type", "cross_validation")
            mlflow.log_param("cv_initial_period", "730 days")
            mlflow.log_param("cv_period", "180 days")
            mlflow.log_param("cv_horizon", "365 days")
            
            # Log the CV model to MLflow
            model_info_cv = mlflow_prophet.log_model(prophet_model, 
                                                     name=f"prophet_model_cv_{code_station}",
                                                     registered_model_name=f"{REGISTERED_MODEL_NAME}", 
                                                     input_example=train_fbp.drop(columns=["y"]).iloc[:5])
            
            # Save serialized CV model
            prophet_model_cv_pickle_path = serialized_model_dir / f"prophet_model_cv_{code_station}.pkl"
            prophet_model_cv_joblib_path = serialized_model_dir / f"prophet_model_cv_{code_station}.joblib"
            export_model_with_pickle(prophet_model, str(prophet_model_cv_pickle_path))
            export_model_with_joblib(prophet_model, str(prophet_model_cv_joblib_path))
            
            # Log CV results and metrics as CSV files
            cv_results_csv_path = local_artifact_dir / f"cv_results_{FILE_POSTFIX_NAME}_{code_station}.csv"
            cv_metrics_csv_path = local_artifact_dir / f"cv_metrics_{FILE_POSTFIX_NAME}_{code_station}.csv"
            cv_results.to_csv(cv_results_csv_path, index=False)
            metrics.to_csv(cv_metrics_csv_path, index=False)
            
            # Log CV results to MLflow
            mlflow.log_artifact(str(cv_results_csv_path))
            mlflow.log_artifact(str(cv_metrics_csv_path))
            
            # Log CV model summary
            print(f"[INFO] Cross-Validation Model logged")
            print(f"[INFO] CV Run ID: {cv_run.info.run_id}")
            print(f"Mean CV Metrics:")
            for metric_name, metric_value in cv_mean_metrics.items():
                print(f"  cv_{metric_name}: {metric_value:.4f}")

        mlflow.log_param("cv_child_run_id", cv_run.info.run_id)
        
        # ============================================================================================
        # IX) Parent-level Registration, Tagging, and Final Artifact Logging
        # ============================================================================================
        
        # Now that both nested runs are complete, register the model at the parent level
        # The model_info from the baseline run contains the model_uri needed for registration
        registered_mv = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=REGISTERED_MODEL_NAME,
        )
        
        model_version = int(registered_mv.version)
        print(f"\n[INFO] Model logged as version {model_version}")

        # Set alias for easy model lookup
        client.set_registered_model_alias(name=REGISTERED_MODEL_NAME,
                                        alias=MODEL_ALIAS_NAME,
                                        version=str(model_version),
                                    )
        print(f"[INFO] Alias '{MODEL_ALIAS_NAME}' now points to version {model_version}")

        # Tag the model version with baseline metrics
        client.set_model_version_tag(name=REGISTERED_MODEL_NAME,
                                    version=str(model_version),
                                    key="dataset",
                                    value="Hubeau Observations Elaborated Dataset",
                                )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=str(model_version),
            key="baseline_test_mape",
            value=f"{mape:.4f}",
        )
        
        # Tag the model version with cross-validation metrics summary
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=str(model_version),
            key="cv_mean_rmse",
            value=f"{cv_mean_metrics['rmse']:.4f}",
        )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=str(model_version),
            key="cv_mean_mape",
            value=f"{cv_mean_metrics['mape']:.4f}",
        )
        print(f"[INFO] Model version {model_version} tagged with both baseline and CV metrics")
        
        # Log all plots and data files to parent run artifacts
        plots_files = [f for f in os.listdir(local_artifact_dir) 
                      if f.endswith((".html", ".png")) and "artifact" not in f]
        for file in plots_files:
            file_path = local_artifact_dir / file
            if file_path.is_file():
                mlflow.log_artifact(str(file_path), artifact_path="plots")
    
    # Transition to production
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        # version=str(model_info.registered_model_version),
        version=str(model_version),
        stage="Production",
    )
    
    print("...Done!")
    print(f"--- Total training time: {time.time() - start_time:.2f} seconds")
    
    # delete local artifact directory after logging to MLflow to save disk space (optional)
    # shutil.rmtree(local_artifact_dir)
    return prophet_model, forecast, prophet_metrics



if __name__ == "__main__":
    main_train_prophet()
