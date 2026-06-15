# -*- coding=utf-8 -*-
# Import necessary modules for path manipulation and standard OS operations
from pathlib import Path
import os
import time

# Import type hinting functionality
from typing import Any, cast

# Import pandas for data manipulation
import pandas as pd
# Import Plotly for generating interactive graphics
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import MLflow for experiment tracking, model logging, and parameter tracking
import mlflow
import mlflow.sklearn
import mlflow.prophet
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# Import the shared MLflow model registration utility
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.model_utils import register_and_alias_model, plot_prophet_forecast

# Get current directory of this script file
CURRENT_DIR = Path(__file__).resolve().parent
# Create directory Plot if not exist
Plot_DIR = CURRENT_DIR / "Plot"
Plot_DIR.mkdir(exist_ok=True)

# Import evaluation metrics from scikit-learn
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Import the Prophet model and its diagnostic tools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings

# Filter out non-critical warnings to keep the execution log clean
warnings.filterwarnings("ignore")

def run_prophet(train_fbp: pd.DataFrame, test_fbp: pd.DataFrame, seasonality: bool, changepoint_prior_scale: float, seasonality_prior_scale: float) -> Prophet:
    """
    Initializes and fits a Prophet model using the provided training data and parameters.
    """
    # Define the baseline growth model type for Prophet
    growth = "linear"
    # Prophet requires boolean flags for seasonality components, we cast to Any to avoid type checking errors
    enabled_seasonality = cast(Any, True)
    
    # Initialize the Prophet model based on whether automatic seasonality is requested
    if seasonality:
        # Instantiate a Prophet model with default auto-seasonality
        prophet_model = Prophet(growth=growth)
    else:
        # Check if the user provided custom hyperparameter tuning values
        if changepoint_prior_scale != 0.05 and seasonality_prior_scale != 10:
            # Instantiate Prophet with explicit custom scales and enabled seasonality components
            prophet_model = Prophet(growth=growth,
                                    changepoint_prior_scale=changepoint_prior_scale,
                                    seasonality_prior_scale=seasonality_prior_scale,
                                    yearly_seasonality=enabled_seasonality,
                                    weekly_seasonality=enabled_seasonality,
                                    daily_seasonality=enabled_seasonality)
        else:
            # Instantiate Prophet with default scales but explicitly enabled seasonality components
            prophet_model = Prophet(growth=growth,
                                    daily_seasonality=enabled_seasonality, 
                                    weekly_seasonality=enabled_seasonality, 
                                    yearly_seasonality=enabled_seasonality)

    # Fit the initialized Prophet model onto the training dataset
    prophet_model.fit(train_fbp)
    
    # Return the trained model object
    return prophet_model

def compute_prophet_metrics(test_fbp: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """
    Computes standard regression metrics comparing the Prophet forecast against the test dataset.
    """
    # Merge the actual test observations with the model's predicted values (yhat) on the date column (ds)
    merged = pd.merge(test_fbp, forecast[["ds", "yhat"]], on="ds", how="inner")
    
    # Initialize an empty dictionary to hold the computed metric values
    prophet_metrics = {}
    
    # Verify that the merge resulted in overlapping data points to evaluate
    if merged.empty:
        print("No overlapping dates between forecast and test set.")
    else:
        # Extract the actual values and the predicted values into separate variables
        y_actual, y_predicted = merged["y"], merged["yhat"]
        
        # Calculate regression metrics and populate the dictionary
        prophet_metrics = {
            "std_mape": mean_absolute_percentage_error(y_actual, y_predicted), # Mean Absolute Percentage Error
            "std_r2": r2_score(y_actual, y_predicted),                         # R-squared value
            "std_mse": mean_squared_error(y_actual, y_predicted),              # Mean Squared Error
            "std_rmse": root_mean_squared_error(y_actual, y_predicted),        # Root Mean Squared Error
            "std_mae": mean_absolute_error(y_actual, y_predicted)              # Mean Absolute Error
        }
        
    # Return the dictionary of calculated metrics
    return prophet_metrics

def main_train_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                       site_name: str, code_station:str, code_site:str, site_measure:str, 
                       SPLIT_DATE: str, TITLE_POSTFIX_NAME: str, 
                       seasonality=False, changepoint_prior_scale=0.05, seasonality_prior_scale=10) -> tuple[Prophet, pd.DataFrame, dict]:
    """
    The main execution function for the Prophet model.
    It manages MLflow tracking, trains the model, generates the forecast, computes metrics, and logs artifacts.
    """
    
    # Define MLflow Registry variables
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_TRACKING_URI")
    REGISTERED_MODEL_NAME = "flood_forecast_model_prophet"
    MODEL_ALIAS_NAME = "challenge"
    
    # Set the URIs
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)

    # Define a standard MLflow experiment name for Prophet executions
    MLFLOW_EXPERIMENT_NAME = "final_project_forecasting_hubeau/prophet"
    
    # Check if the MLflow experiment exists in the tracking server
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        # Create the experiment if it does not currently exist
        mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
        
    # Set the current active MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Start a new MLflow run with a specific run name tying it to the current station
    with mlflow.start_run(run_name=f"prophet_station_{code_station}") as run:
        
        # Log metadata parameters to the MLflow run
        mlflow.log_param("site_name", site_name)
        mlflow.log_param("code_station", code_station)
        mlflow.log_param("data_split_date", SPLIT_DATE)
        
        # Log model hyperparameters to the MLflow run
        mlflow.log_param("seasonality", seasonality)
        mlflow.log_param("changepoint_prior_scale", changepoint_prior_scale)
        mlflow.log_param("seasonality_prior_scale", seasonality_prior_scale)
        
        # Record the start time to monitor training duration
        start_time = time.time()
        print(f"[{start_time}] Training baseline Prophet model...")
        
        # Trigger the model training process
        prophet_model = run_prophet(train_df, test_df, seasonality, changepoint_prior_scale, seasonality_prior_scale)
        
        # Generate a future dataframe that spans both the training timeframe and extends into the test timeframe
        future = prophet_model.make_future_dataframe(periods=len(test_df))
        
        # Use the trained model to predict values over the entire defined timeframe
        prophet_forecast = prophet_model.predict(future)
        
        # Evaluate the prediction accuracy using the test set
        prophet_metrics = compute_prophet_metrics(test_df, prophet_forecast)

        # Generate Plotly graphic only for Prophet Forecast against actual Train/Test sets.
        prophet_fig = plot_prophet_forecast(train_df, test_df, prophet_forecast, TITLE_POSTFIX_NAME, site_measure)
        
        # Save the figure to an HTML file
        prophet_fig.write_html(str(Plot_DIR / f"02_prophet_forecast_plot_{code_station}.html"))

        # Iterate over the calculated metrics and log each one natively to MLflow
        for k, v in prophet_metrics.items():
            mlflow.log_metric(k, v)
            
        # Print the metrics to the console for real-time review
        print("Prophet metrics:", prophet_metrics)
        
        # Generate the standard Prophet time-series components plot (trend, seasonality)
        fig3 = prophet_model.plot_components(prophet_forecast)
        
        # Log the components plot directly into MLflow as an artifact, organized in a Plot directory
        fig3.savefig(str(Plot_DIR / f"03_prophet_components_plot_{code_station}.png"))

        # Log the figures to MLflow as an HTML artifact for dashboard visualization
        mlflow.log_artifact(str(Plot_DIR))
        
        # Tag for deployment tracking
        mlflow.set_tags({
            "model_type": "prophet",
            "project": "flood_forecast",
            "station_id": code_station,
            "site_name": site_name,
            "site_measure": site_measure,
            "code_site": code_site,
            "dataset": "hubeau_api"
        })

        # Infer the model signature from the training features and predictions
        signature = infer_signature(
            train_df.drop(columns=["y"]), 
            prophet_model.predict(train_df.drop(columns=["y"]))
        )

        # Log the baseline model to MLflow with the signature and input example
        model_info = mlflow.prophet.log_model(
            pr_model=prophet_model, 
            name=f"prophet_model_{code_station}",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature, 
            input_example=train_df.drop(columns=["y"]).iloc[:5]
        )
        
        # Register and alias model
        register_and_alias_model(
            model_uri=model_info.model_uri,
            registered_model_name=REGISTERED_MODEL_NAME,
            alias_name=MODEL_ALIAS_NAME,
            tracking_uri=MLFLOW_TRACKING_URI,
            registry_uri=MLFLOW_REGISTRY_URI
        )
        
        # Return the model, its forecast, and the computed metrics to be used by downstream models in the pipeline
        return prophet_model, prophet_forecast, prophet_metrics
