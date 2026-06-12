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
#
import mlflow
import mlflow.xgboost

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient  # Import MlflowClient to interact with the MLflow tracking server

# ============================================================
# Necessary modules for Scikit-learn
# ============================================================
# Import train_test_split and TimeSeriesSplit for splitting the data, and various metrics for evaluating the model"s performance
from sklearn.model_selection import train_test_split, TimeSeriesSplit
# Import des métriques sklearn pour l'évaluation
from sklearn.metrics import  (r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error)

# ============================================================
# Necessary modules for XGBOOST
# ============================================================
# Librairie les modèles de machine learning ensemblistes
import xgboost as xgb
from xgboost import XGBRegressor  # Import XGBoost regressor model.

# Import joblib and pickle for model serialization
import pickle
import joblib

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


def plot_time_series_with_plotly(clean_data_df: pd.DataFrame, STATION_MEASURE: str, TITLE_POSTFIX_NAME: str) -> None:
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


def plot_prophet_components(prophet_model: Prophet, test_fcst: pd.DataFrame, local_artifact_dir: Path, FILE_POSTFIX_NAME: str, TITLE_POSTFIX_NAME: str) -> None:
    # Plot the components of the model using the built-in Prophet function    
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
    # save the figure as an HTML and png file
    fig4.write_html(local_artifact_dir / f"04_prophet_components_plot_{FILE_POSTFIX_NAME}.html")
    # fig4.write_image(local_artifact_dir / f"04_prophet_components_plot_{FILE_POSTFIX_NAME}.png")
    # fig4.show()
    return fig4

def plot_forecast_with_plotly(prophet_model, forecast: pd.DataFrame, TITLE_POSTFIX_NAME: str, FILE_POSTFIX_NAME: str, local_artifact_dir: Path):
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
    # save the figure as an HTML and png file
    fig6.write_html(local_artifact_dir / f"06_prophet_forecast_plot_{FILE_POSTFIX_NAME}.html")
    # fig6.write_image(local_artifact_dir / f"06_prophet_forecast_plot_{FILE_POSTFIX_NAME}.png")
    # fig6.show()
    return fig6


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


## Feature Creation
def create_features(df):
    """
    Create time series features based on time series index.
    Arguments:
        df: pandas dataframe with datetime index
    Returns:
        df: pandas dataframe with new features added
    """
    df = df.copy()
    # create time series features based on datetime index
    
    # check if the index is a datetime index, if not, try to convert it to datetime
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour  # add hour of the day as a feature: 0-23 
        df['dayofweek'] = df.index.dayofweek  # add day of the week as a feature: 0-6
        df['quarter'] = df.index.quarter  # add quarter of the year as a feature: 1-4
        df['month'] = df.index.month  # add month as a feature: 1-12
        df['year'] = df.index.year  # add year as a feature
        df['dayofyear'] = df.index.dayofyear  # add day of the year as a feature: 1-365
        df['dayofmonth'] = df.index.day  # add day of the month as a feature: 1-31
        df['weekofyear'] = df.index.isocalendar().week  # add week of the year as a feature: 1-52
    else:
        df['hour'] = df["ds"].dt.hour  # add hour of the day as a feature: 0-23 
        df['dayofweek'] = df["ds"].dt.dayofweek  # add day of the week as a feature: 0-6
        df['quarter'] = df["ds"].dt.quarter  # add quarter of the year as a feature: 1-4
        df['month'] = df["ds"].dt.month  # add month as a feature: 1-12
        df['year'] = df["ds"].dt.year  # add year as a feature
        df['dayofyear'] = df["ds"].dt.dayofyear  # add day of the year as a feature: 1-365
        df['dayofmonth'] = df["ds"].dt.day  # add day of the month as a feature: 1-31
        df['weekofyear'] = df["ds"].dt.isocalendar().week  # add week of the year as a feature: 1-52
    return df


def plot_train_test_split(xgb_train_df, xgb_test_df, SPLIT_DATE, site_name, STATION_MEASURE):
    # Visualize the train/test split using plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xgb_train_df.index, y=xgb_train_df['y'], mode='lines', name='Train Set'))
    fig.add_trace(go.Scatter(x=xgb_test_df.index, y=xgb_test_df['y'], mode='lines', name='Test Set'))
    fig.add_vline(x=SPLIT_DATE, line_dash='dash', line_color='black')
    # fig.add_vline(x=SPLIT_DATE, line_dash="dash", line_color="red", annotation_text="Train-Test split line", annotation_position="top left")

    # Customize the layout
    # fig.update_layout(title='Data Train/Test Split', xaxis_title='Date', yaxis_title='Water Height', width=1200, height=600)
    fig.update_layout(title=f'XGB : Data Train/Test Split - {site_name} : Water Height Over Time: {STATION_MEASURE}',
                    title_x=0.5,
                    title_font=dict(size=24, color='black', family='Arial', weight='bold'),
                    xaxis_title='Datetime', 
                    yaxis_title=f'Water Height ({STATION_MEASURE})',
                    legend_title='Legend',
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='black', borderwidth=1),
                    # Size of the figure
                    width=1200, height=600
    )
    # fig.show()
    return fig

# *============================================================
# *Implementation of XGBOOST (START)
# *============================================================

def run_xgboost(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, random_state=42) -> XGBRegressor and XGBRegressor:
    # xgb_regressor_model = XGBRegressor(n_estimators=100,
    #                                    eval_metric="rmse",
    #                      max_depth=3,
    #                      n_jobs=4,
    #             )
    # xgb_regressor_model.fit(X_train, y_train)

    # Create booster instance of ML Model using XGBoost
    xgb_regressor_class = XGBRegressor(base_score=0.5,  # default base score = 0.5, global bias
                            booster='gbtree',    # default booster = gbtree, tree-based models are usually better for tabular data    
                            n_estimators=1000,   # number of trees to fit, large number for early stopping to work
                            early_stopping_rounds=50,    # stop if no improvement after 50 rounds of training
                            objective='reg:linear',  # regression task: predict a continuous value -> this is the default for XGBRegressor, so we could have omitted it
                            max_depth=3, # maximum depth of each tree, controls model complexity and overfitting
                            learning_rate=0.01,  # step size shrinkage used to prevent overfitting, smaller values lead to more robust models but require more trees (n_estimators)
                            n_jobs=2,  # use CPU cores. Note: set -1 for all CPU cores for training
                            eval_metric='rmse',  # evaluation metric to use for early stopping, root mean squared error in this case
                            eval_names=['train', 'test'],  # display names for the eval sets in the training logs
                            random_state=random_state  # set random seed for reproducibility
                        )

    # Fit gradient boosting model: fit the model with early stopping
    xgb_regressor_model = xgb_regressor_class.fit(X=X_train,  # Input feature matrix: it's training data features X_train (fit the model to the training data)
                                        y=y_train,   # Labels/Target values: it's training data labels y_train (fit the model to the training data)
            eval_set=[(X_train, y_train), (X_test, y_test)],    # A list of (X, y) tuple pairs to use as validation sets, for which metrics will be computed: this will evaluate performance on both training and test sets during training
            verbose=100,    # print evaluation results every 100 trees:  If verbose is True and an evaluation set is used, the evaluation metric measured on the validation set is printed to stdout at each boosting stage. If verbose is an integer, the evaluation metric is printed at each verbose boosting stage. The last boosting stage / the boosting stage found by using early_stopping_rounds is also printed.
        )
    return xgb_regressor_class, xgb_regressor_model


# *============================================================
# *Implementation of XGBOOST (END)
# *============================================================

def plot_xgboost_predictions(y_test, y_test_pred, site_name, target_measure = "Water Height",):
    # Display the predictions vs true values using plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='markers', name='True Values'))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test_pred, mode='markers', name='Predicted Values'))

    # fig.add_trace(go.Scatter(x=y_test.index, y=y_test_pred, mode='lines', name='True Values', line=dict(color='cyan')))
    # fig.add_trace(go.Scatter(x=y_test.index, y=y_test_pred, mode='lines', name='XGBoost Forecast', line=dict(color='cyan')))

    fig.update_layout(title=f'XGBoost - {site_name} : True vs Predicted Values for {target_measure} (HIXnJ)',
                    title_x=0.5,
                    xaxis_title='Date',
                    yaxis_title=target_measure,
                    width=1400, height=600
                    )
    # fig.show()
    return fig


def figure_XGB(y_test, y_test_pred, train_fbp, test_fbp, prophet_forecast, target_measure = "Water Height", TITLE_POSTFIX_NAME="") -> go.Figure:
    # Plot the forecast with confidence intervals and actual history using plotly with interactive plot and range slider options
    # We visualize the forecast, confidence intervals, and actual history.
    # Use plotly only to visualize the forecast, confidence intervals, and actual history with interactive plot and range slider options
    fig_xgb = go.Figure()
    # Add the actual training data
    fig_xgb.add_trace(go.Scatter(x=train_fbp['ds'], y=train_fbp['y'], mode='lines+markers', name='Actual (Training)', line=dict(color='blue')))
    # Add the actual test data
    fig_xgb.add_trace(go.Scatter(x=test_fbp['ds'], y=test_fbp['y'], mode='lines+markers', name='Actual (Test)', line=dict(color='green')))
    # Add the forecast line
    if prophet_forecast is not None:
        fig_xgb.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red')))
        # Add confidence interval upper bound (invisible line to define fill boundary)
        fig_xgb.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'], fill=None, mode='lines', 
                                line_color='rgba(0,0,0,0)', showlegend=False))
        # Add confidence interval lower bound with fill to upper bound
        fig_xgb.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'], fill='tonexty', mode='lines', 
                                line_color='rgba(0,0,0,0)', name='Confidence Interval', 
                                fillcolor='rgba(255,0,0,0.2)'))

    # XGBoost forecast line
    fig_xgb.add_trace(go.Scatter(x=y_test.index, y=y_test_pred, mode='lines', name='XGBoost Forecast', line=dict(color='cyan')))

    # add horizontal line for flood threshold (example: 2.86) in April 1983
    fig_xgb.add_hline(y=2.86*1000, line_dash="dash", line_color="red", annotation_text=f"Flood Threshold ({2.86*1000}) in April 1983", annotation_position="top left")
    # add horizontal line for flood threshold (example: 2.77) in February 1990
    fig_xgb.add_hline(y=2.77*1000, line_dash="dash", line_color="cyan", annotation_text=f"Flood Threshold ({2.77*1000}) in February 1990", annotation_position="bottom left")
    # add horizontal line for flood threshold (example: 2.58) in January 2018
    fig_xgb.add_hline(y=2.58*1000, line_dash="dash", line_color="green", annotation_text=f"Flood Threshold ({2.58*1000}) in January 2018", annotation_position="bottom left")

    # Update layout with range selector buttons and range slider
    fig_xgb.update_layout(
        title=f"XGBoost - Prophet Forecast | XGBoost - {target_measure} Over Time - {TITLE_POSTFIX_NAME}",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title=target_measure,
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
    return fig_xgb


# run script with arguments example: python test.py --test-size 0.25 --random-state 123
# for comparison with Prophet set <prophet_forecast> which is an optional argument that can be passed to the main_train_xgboost function if you want to include the Prophet forecast in the XGBoost forecast plot. 
# If you have already trained a Prophet model and generated a forecast, you can pass that forecast as a DataFrame to this function to visualize it alongside the XGBoost 
# predictions. If you do not have a Prophet forecast or do not want to include it in the plot, you can simply call main_train_xgboost() without any arguments, 
# and it will proceed with training the XGBoost model and plotting the results without the Prophet forecast.
def main_train_xgboost(prophet_forecast:Prophet = None) -> dict:
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
    MLFLOW_EXPERIMENT_NAME = "final_project_forecasting_hubeau/xgboost"
    REGISTERED_MODEL_NAME = "flood_forecast_model_xgboost"
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
    
    fig1 = plot_time_series_with_plotly(clean_data_df, site_measure, TITLE_POSTFIX_NAME)

    # =======================================================================
    # II) Train/Test Split of the data for time series forecasting with Prophet
    # =======================================================================
    
    TRAIN_SIZE_RATIO = 0.8  # 80% of the data will be used for training, and the remaining 20% will be used for testing.
    train_fbp, test_fbp, SPLIT_DATE = split_train_test(clean_data_df, train_ratio=TRAIN_SIZE_RATIO, split_by_ratio=True)  # Split data into train and test sets.

    # save the train and test sets as csv files
    train_fbp.to_csv(local_artifact_dir / f"train_fbp_{FILE_POSTFIX_NAME}.csv", index=False)
    test_fbp.to_csv(local_artifact_dir / f"test_fbp_{FILE_POSTFIX_NAME}.csv", index=False)
    
    print(clean_data_df.head())
    print(clean_data_df.tail())
    
    
    # *============================================================
    # *============================================================
    # Define the column names for the datetime and target variable in the dataset
    # Data preparation for XGBoost: we need to create features from the datetime column and ensure the target variable is properly defined for regression.
    # xgb_data_df = clean_data_df.copy()
    # xgb_data_df = xgb_data_df.sort_values("ds") # sort by datetime column to ensure proper feature creation
    # # set the datetime column as the index of the dataframe
    # xgb_data_df = xgb_data_df.set_index("ds")

    # Create features for the entire dataset
    xgb_data_df = create_features(clean_data_df)

    # ## Train / Test Split for XGBoost
    # # *==== Method 2 ====:  Proper Data Splitting
    # # CRITICAL: Use chronological split, NOT random split!

    # # size of the training set is 80% of the entire dataset, and the remaining 20% is used for testing.
    # # train_size = int(len(xgb_data_df) * TRAIN_SIZE_RATIO)
    # # xgb_train_df = xgb_data_df[:train_size]
    # # xgb_test_df = xgb_data_df[train_size:]
    # xgb_train_df, xgb_test_df, xgb_SPLIT_DATE = split_train_test(clean_data_df, train_ratio=TRAIN_SIZE_RATIO, split_by_ratio=True)  # Split data into train and test sets.
    # print(f"Split date : {xgb_SPLIT_DATE},\t Train: {len(xgb_train_df)},\t Test: {len(xgb_test_df)}")
    # print(xgb_train_df.head())
    # print(xgb_test_df.head())
    
    # *============================================================
    # *============================================================
    
    print(xgb_data_df.columns.tolist())
    
    # new features names for Regression
    feature_names = xgb_data_df.columns.tolist()
    feature_names.remove("y")

    # remove empty columns from features: we remove "hour" in this case because it has the same value (0) for all rows, which can cause issues for XGBoost training
    for col in ["ds", "hour"]:
        if col in feature_names:
            feature_names.remove(col)
    print("Features utilisées :", feature_names)

    # *split train/test in features (X) and target variable (y)
    X = xgb_data_df[feature_names]  # features: water height and flow rate
    y = xgb_data_df["y"]          # target: water height
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    print("\n" + "="*100)
    print("Starting XGBoost evaluation in nested run...")
    print("="*100 + "\n")    
    # ===========================================================================================
    # Start a parent MLflow run for the entire experiment to organize both baseline and CV models
    # as nested runs for clearer comparison and organization in the MLflow UI.
    # ===========================================================================================
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=f"xgb_station_{code_station}") as xgb_run:
        # Log shared experiment-level parameters
        mlflow.log_param("site_name", site_name)
        mlflow.log_param("data_split_date", SPLIT_DATE)
        
        mlflow.log_figure(fig1, f"Plot/01_time_series_plot_{FILE_POSTFIX_NAME}.html")
        # # Visualize the train/test split using plotly
        # plot_train_test_split(xgb_train_df, xgb_test_df, xgb_SPLIT_DATE, site_name, TITLE_POSTFIX_NAME, FILE_POSTFIX_NAME, local_artifact_dir)
        # plot_train_test_split(train_fbp, test_fbp, SPLIT_DATE, site_measure, f"XGBoost_{TITLE_POSTFIX_NAME}", f"XGBoost_{FILE_POSTFIX_NAME}", local_artifact_dir)

        # Log the train/test split visualization and data files as parent-level artifacts
        mlflow.log_artifacts(str(local_artifact_dir))
        
        start_time = time.time()
        print(f"{start_time} -> Training baseline model...")
        # =======================================================================
        # III) XGBoost Model Evaluation
        # =======================================================================

        # *run XGBoost regression    
        xgb_regressor_class, xgb_regressor_model = run_xgboost(X_train, X_test, y_train, y_test)

        # display scores: accuracy
        accuracy = xgb_regressor_model.score(X_test, y_test)
        print(f"\nAccuracy: {accuracy}")

        # get predictions from the fitted model
        y_test_pred = xgb_regressor_model.predict(X_test)
        
        # plot the predictions vs true values using plotly
        pred_df = pd.DataFrame({"ds": test_fbp["ds"].values, "y_true": test_fbp["y"].values, "y_pred": y_test_pred})
        print(pred_df.head())  # Build aligned output.
        
        fig = plot_xgboost_predictions(y_test, y_test_pred, site_name, target_measure=site_measure)
        mlflow.log_figure(fig, artifact_file=f"Plot/02_xgboost_predictions_plot_{FILE_POSTFIX_NAME}.html")
        

        # display scores: r^2, RMSE, MAE
        r2 = xgb_regressor_model.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = root_mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)

        print("R^2 score:", r2)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("MAPE:", mape)

        # ================= fig 7 bis: Compare XGBoost predictions with Prophet forecast and actual history =================
        # prophet_forecast : replace with actual prophet forecast dataframe if available, or keep as None to skip plotting prophet forecast
        fig_xgb = figure_XGB(y_test, y_test_pred, train_fbp, test_fbp, prophet_forecast, target_measure=site_measure, TITLE_POSTFIX_NAME=FILE_POSTFIX_NAME)
        mlflow.log_figure(fig_xgb, artifact_file=f"Plot/03_xgboost_vs_prophet_comparison_plot_{FILE_POSTFIX_NAME}.html")
        
        # Log XGBoost validation metrics
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("run_type", "xgboost_validation")
        xgb_metrics = {
            "xgb_r2": r2,
            "xgb_mse": mse,
            "xgb_rmse": rmse,
            "xgb_mae": mae,
            "xgb_mape": mape,
            "xgb_accuracy": accuracy
        }
        mlflow.log_metrics(xgb_metrics)
        
        # Log XGBoost model parameters
        xgb_params = {param: value for param, value in xgb_regressor_class.get_params().items()}
        mlflow.log_params(xgb_params)
        
        # Log the trained XGBoost model to MLflow and register it in the model registry
        model_info_xgb = mlflow.sklearn.log_model(sk_model=xgb_regressor_model, 
                                name=f"xgboost_model_{code_station}",
                                registered_model_name=REGISTERED_MODEL_NAME,
                                input_example=X_test.iloc[:5],  # Log an input example for the model signature, using the first 5 rows of the test set features
                                signature=infer_signature(X_test, y_test_pred)  # Infer the model signature from the test features and predictions
                            )
        
        # Tag for tracking the training time of the model
        mlflow.set_tags({
            "model_type": "xgboost",
            "project": "flood_forecast",
            "station_id": code_station,
            "site_name": site_name,
            "site_measure": site_measure,
            "code_site": code_site,
            "dataset": "hubeau_api", 
            "framework": "xgboost",
            "training_time_seconds": time.time() - start_time,
            "training_time_minutes": (time.time() - start_time) / 60,
            "training_time_hours": (time.time() - start_time) / 3600
        })
        
        # set registered model description with the main parameters and metrics of the model
        registered_xgb = mlflow.register_model(model_uri=model_info_xgb.model_uri,
                                                name=f"{REGISTERED_MODEL_NAME}", 
                                            )
        
        xgb_model_version = int(registered_xgb.version)
        print(f"\n[INFO] Model logged as version {xgb_model_version}")

        # Set alias for easy model lookup & deployment
        client.set_registered_model_alias(name=f"{REGISTERED_MODEL_NAME}",
                                        alias=MODEL_ALIAS_NAME,
                                        version=str(xgb_model_version)
                                    )
        print(f"[INFO] Alias '{MODEL_ALIAS_NAME}' now points to version {xgb_model_version}")
        
        # Log XGBoost results and metrics as CSV files
        xgb_metrics_csv_path = local_artifact_dir / f"xgboost_metrics_{FILE_POSTFIX_NAME}_{Current_time}.csv"
        xgb_metrics_df = pd.DataFrame([xgb_metrics])
        xgb_metrics_df.to_csv(xgb_metrics_csv_path, index=False)
        mlflow.log_artifact(str(xgb_metrics_csv_path))
        
    # log the run ID of the XGBoost training run as a parameter for easier reference and tracking in MLflow
    mlflow.log_param("xgb_run_id", xgb_run.info.run_id)
    
    xgb_return_dict = {
        "class": xgb_regressor_class,
        "model": xgb_regressor_model,
        "metrics": xgb_metrics,
        "info": model_info_xgb,
        "registered": registered_xgb
    }
    return xgb_return_dict


if __name__ == "__main__":
    main_train_xgboost()
