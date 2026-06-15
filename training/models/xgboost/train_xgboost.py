# -*- coding=utf-8 -*-
# Import necessary modules for path manipulation and standard OS operations
from pathlib import Path
import os
import time

# Import pandas and numpy for data manipulation and array operations
import pandas as pd
import numpy as np
from typing import Any

# Import MLflow for experiment tracking and model logging
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

# Import XGBoost library for gradient boosting regression
import xgboost as xgb
from xgboost import XGBRegressor

# Import evaluation metrics from scikit-learn
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings

# Filter out non-critical warnings to keep the execution log clean
warnings.filterwarnings("ignore")

# Import the feature creation utility and the shared MLflow model registration utility
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.data_utils import create_features
from config.model_utils import register_and_alias_model, plot_xgboost_forecast

# Get current directory of this script file
CURRENT_DIR = Path(__file__).resolve().parent
# Create directory Plot if not exist
Plot_DIR = CURRENT_DIR / "Plot"
Plot_DIR.mkdir(exist_ok=True)

def run_xgboost(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, random_state=42) -> tuple[XGBRegressor, Any]:
    """
    Initializes and fits an XGBoost Regressor model using the provided training features and labels.
    """
    # Instantiate the XGBRegressor with specified hyperparameters for time series forecasting
    xgb_regressor_class = XGBRegressor(
        base_score=0.5,                    # Initial prediction score of all instances, global bias
        booster='gbtree',                  # Specify which booster to use: gbtree, gblinear or dart
        n_estimators=1000,                 # Number of gradient boosted trees. Equivalent to number of boosting rounds.
        early_stopping_rounds=50,          # Activates early stopping. Validation metric needs to improve at least once in every early_stopping_rounds round(s)
        objective='reg:squarederror',      # Specify the learning task and the corresponding learning objective
        max_depth=3,                       # Maximum tree depth for base learners.
        learning_rate=0.01,                # Boosting learning rate (xgb's "eta")
        n_jobs=2,                          # Number of parallel threads used to run xgboost
        eval_metric='rmse',                # Evaluation metrics for validation data
        random_state=random_state          # Random number seed
    )

    # Fit the initialized XGBoost model onto the training dataset, evaluating on both train and test sets
    xgb_regressor_model = xgb_regressor_class.fit(
        X=X_train,                                       # Training feature matrix
        y=y_train,                                       # Training labels
        eval_set=[(X_train, y_train), (X_test, y_test)], # A list of (X, y) tuple pairs to use as validation sets
        verbose=100                                      # Print evaluation metrics every 100 boosting stages
    )
    
    # Return both the class instance and the fitted model object
    return xgb_regressor_class, xgb_regressor_model

def main_train_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                       site_name: str, code_station: str, code_site:str, site_measure:str, 
                       SPLIT_DATE: str, prophet_forecast: pd.DataFrame = None, random_state=42, TITLE_POSTFIX_NAME: str="") -> pd.Series:
    """
    The main execution function for the XGBoost model.
    It manages MLflow tracking, feature extraction, model training, metric computation, and logging.
    """
    
    # Define MLflow Registry variables
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_TRACKING_URI")
    REGISTERED_MODEL_NAME = "flood_forecast_model_xgboost"
    MODEL_ALIAS_NAME = "challenge"
    
    # Set the URIs
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)

    # Define a standard MLflow experiment name for XGBoost executions
    MLFLOW_EXPERIMENT_NAME = "final_project_forecasting_hubeau/xgboost"
    
    # Check if the MLflow experiment exists in the tracking server
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        # Create the experiment if it does not currently exist
        mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
        
    # Set the current active MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Set the newly created or existing experiment as the active one for this script
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"xgb_station_{code_station}") as xgb_run:
        # Log dataset information
        mlflow.log_param("dataset", "hubeau_api")
        mlflow.log_param("code_station", code_station)
        mlflow.log_param("data_split_date", SPLIT_DATE)
        mlflow.log_param("random_state", random_state)
        
        # Record the start time to monitor training duration
        start_time = time.time()
        print(f"[{start_time}] Training XGBoost model...")
        
        # Extract datetime features (like hour, day of week, month, etc.) from the date column for the training set
        train_feat = create_features(train_df)
        
        # Extract datetime features from the date column for the testing set
        test_feat = create_features(test_df)
        
        # Ensure that the 'ds' (datetime) column is set as the index for proper chronological alignment
        if 'ds' in train_feat.columns:
            train_feat = train_feat.set_index('ds')
        if 'ds' in test_feat.columns:
            test_feat = test_feat.set_index('ds')
            
        # Separate the features (X) from the target variable (y) for the training set
        X_train = train_feat.drop(columns=['y'])
        y_train = train_feat['y']
        
        # Separate the features (X) from the target variable (y) for the testing set
        X_test = test_feat.drop(columns=['y'])
        y_test = test_feat['y']
        
        # Trigger the model training process using the separated datasets
        xgb_class, xgb_model = run_xgboost(X_train, X_test, y_train, y_test, random_state)
        
        # Concatenate the training and testing features to generate a complete forecast timeline
        all_X = pd.concat([X_train, X_test])
        
        # Use the trained model to predict values over the entire defined timeframe
        all_y_pred = xgb_model.predict(all_X)
        
        # Convert the resulting numpy array of predictions into a pandas Series with the correct datetime index
        all_y_pred_series = pd.Series(all_y_pred, index=all_X.index)
        
        # Generate predictions specifically for the test set to evaluate model accuracy
        y_test_pred = xgb_model.predict(X_test)
        
        # Calculate regression metrics by comparing the test set predictions against the actual test set values
        metrics = {
            "std_mape": mean_absolute_percentage_error(y_test, y_test_pred), # Mean Absolute Percentage Error
            "std_r2": r2_score(y_test, y_test_pred),                         # R-squared value
            "std_mse": mean_squared_error(y_test, y_test_pred),              # Mean Squared Error
            "std_rmse": root_mean_squared_error(y_test, y_test_pred),        # Root Mean Squared Error
            "std_mae": mean_absolute_error(y_test, y_test_pred)              # Mean Absolute Error
        }

        # Generate Plotly graphic only for XGBoost Forecast against actual Train/Test sets.
        xgb_fig = plot_xgboost_forecast(train_df, test_df, all_y_pred_series, TITLE_POSTFIX_NAME, site_measure)
        
        # Save the figure to an HTML file
        xgb_fig.write_html(str(Plot_DIR / f"02_xgboost_forecast_plot_{code_station}.html"))
        
        # Log the figures to MLflow as an HTML artifact for dashboard visualization
        mlflow.log_artifact(str(Plot_DIR))
        
        # Iterate over the calculated metrics and log each one natively to MLflow
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            
        # Print the metrics to the console for real-time review
        print("XGBoost metrics:", metrics)
        
        # Log XGBoost model parameters
        xgb_params = {param: value for param, value in xgb_class.get_params().items()}
        mlflow.log_params(xgb_params)
        
        mlflow.set_tags({
            "model_type": "xgboost",
            "project": "flood_forecast",
            "station_id": code_station,
            "site_name": site_name,
            "site_measure": site_measure,
            "code_site": code_site,
            "dataset": "hubeau_api", 
            "framework": "scikit-learn",
            "training_time_seconds": time.time() - start_time,
            "training_time_minutes": (time.time() - start_time) / 60,
            "training_time_hours": (time.time() - start_time) / 3600
        })

        # Log the trained XGBoost model to MLflow and register it in the model registry
        model_info = mlflow.xgboost.log_model(
            xgb_model=xgb_model, 
            name=f"xgboost_model_{code_station}",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_test.iloc[:5],
            signature=infer_signature(X_test, xgb_model.predict(X_test))
        )
        
        # Register and alias model
        register_and_alias_model(
            model_uri=model_info.model_uri,
            registered_model_name=REGISTERED_MODEL_NAME,
            alias_name=MODEL_ALIAS_NAME,
            tracking_uri=MLFLOW_TRACKING_URI,
            registry_uri=MLFLOW_REGISTRY_URI
        )
        
        # Return the comprehensive forecast series to be used by the pipeline orchestrator
        return all_y_pred_series
