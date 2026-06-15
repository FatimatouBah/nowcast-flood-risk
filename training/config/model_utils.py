# -*- coding=utf-8 -*-
import argparse
from datetime import date
import pandas as pd
import plotly.graph_objects as go
import os
import mlflow
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MLflow + scikit-learn training script")
    parser.add_argument("--site_name", type=str, default="Kogenheim", help="Site name")
    parser.add_argument("--code_site", type=str, default="A2360030", help="Site code")
    parser.add_argument("--code_station", type=str, default="A236003001", help="Station code")
    parser.add_argument("--site_measure", type=str, default="HIXnJ", help="Site measure")
    parser.add_argument("--start_date", type=str, default="2007-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default=str(date.today()), help="End date")
    
    parser.add_argument("--seasonality", type=bool, default=False, help="Include seasonality components (True or False)")
    parser.add_argument("--changepoint_prior_scale", type=float, default=0.05, help="Changepoint prior scale")
    parser.add_argument("--seasonality_prior_scale", type=float, default=10, help="Seasonality prior scale")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--ncpus", type=int, default=1, help="Number of CPUs")
    return parser.parse_args()


def setup_mlflow():
    """Load environment variables and set MLflow tracking/registry URIs."""
    load_dotenv()
    os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "False"
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_registry_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_registry_uri(mlflow_registry_uri)
    return mlflow_tracking_uri, mlflow_registry_uri


def register_and_alias_model(model_uri: str, registered_model_name: str, alias_name: str, tracking_uri: str, registry_uri: str):
    """
    Registers a logged MLflow model into the Model Registry, sets an alias on the new version,
    and transitions its stage to Production.
    """
    from mlflow.tracking import MlflowClient
    
    # Initialize the MLflow client
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    
    # Register the model
    registered_mv = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )
    
    model_version = int(registered_mv.version)
    print(f"\n[INFO] Model logged as version {model_version}")

    # Set alias for easy model lookup (e.g., "challenge")
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias_name,
        version=str(model_version),
    )
    print(f"[INFO] Alias '{alias_name}' now points to version {model_version}")

    # Transition to production stage
    client.transition_model_version_stage(
        name=registered_model_name,
        version=str(model_version),
        stage="Production",
    )
    print(f"[INFO] Model transitioned to Production stage")


def plot_time_series_with_plotly(clean_data_df: pd.DataFrame, STATION_MEASURE: str, TITLE_POSTFIX_NAME: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clean_data_df["ds"], y=clean_data_df["y"], mode="lines", name=f"Water Height ({STATION_MEASURE})"))
    fig.update_traces(line_color="blue", name=f"Water Height ({STATION_MEASURE})", showlegend=True)
    fig.add_hline(y=2.86*1000, line_dash="dash", line_color="red", annotation_text="Flood Threshold (2.86) in April 1983", annotation_position="top left")
    fig.add_hline(y=2.77*1000, line_dash="dash", line_color="cyan", annotation_text="Flood Threshold (2.77) in February 1990", annotation_position="bottom left")
    fig.add_hline(y=2.58*1000, line_dash="dash", line_color="green", annotation_text="Flood Threshold (2.58) in January 2018", annotation_position="bottom left")
    
    fig.update_layout(
        title=f"{TITLE_POSTFIX_NAME} : Water ({STATION_MEASURE}) Over Time",
        title_x=0.5,
        title_font=dict(size=24, color="black", family="Arial", weight="bold"),
        xaxis_title="Datetime", 
        yaxis_title=f"Water ({STATION_MEASURE})",
        legend_title="Legend",
        xaxis=dict(
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
            rangeslider=dict(visible=True),
            type="date"
        ),
        width=1200, height=600
    )
    return fig


def plot_train_test_split(train_df: pd.DataFrame, test_df: pd.DataFrame, SPLIT_DATE: str, STATION_MEASURE: str, TITLE_POSTFIX_NAME: str) -> go.Figure:
    fig = go.Figure()
    
    if 'ds' in train_df.columns:
        x_train, y_train = train_df["ds"], train_df["y"]
        x_test, y_test = test_df["ds"], test_df["y"]
    else:
        x_train, y_train = train_df.index, train_df.iloc[:, 0]
        x_test, y_test = test_df.index, test_df.iloc[:, 0]

    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode="lines", name="Train Set", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode="lines", name="Test Set", line=dict(color='green')))
    fig.add_vline(x=SPLIT_DATE, line_dash="dash", line_color="black")
    
    fig.update_layout(
        title=f"Data Train/Test Split - Water ({STATION_MEASURE}) Over Time - {TITLE_POSTFIX_NAME}",
        title_x=0.5,
        title_font=dict(size=24, color="black", family="Arial", weight="bold"),
        xaxis_title="Datetime", 
        yaxis_title=f"Water ({STATION_MEASURE})",
        legend_title="Legend",
        width=1200, height=600
    )
    return fig


def plot_all_models_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                               prophet_forecast: pd.DataFrame, 
                               xgb_pred: pd.Series, 
                               tf_lstm_pred: pd.Series, 
                               TITLE_POSTFIX_NAME: str, STATION_MEASURE: str) -> go.Figure:
    """Generate a combined Plotly graphic for all models against actual Train/Test sets."""
    fig = go.Figure()
    
    # Actuals
    x_train, y_train = (train_df["ds"], train_df["y"]) if 'ds' in train_df.columns else (train_df.index, train_df.iloc[:, 0])
    x_test, y_test = (test_df["ds"], test_df["y"]) if 'ds' in test_df.columns else (test_df.index, test_df.iloc[:, 0])
    
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode="lines", name="Actual (Training)", line=dict(color="blue", width=1.5)))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode="lines", name="Actual (Test)", line=dict(color="green", width=1.5)))
    
    # Prophet Forecast
    if prophet_forecast is not None:
        fig.add_trace(go.Scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat"], mode="lines", name="Prophet Forecast", line=dict(color="red", width=2)))

    # XGBoost Forecast
    if xgb_pred is not None:
        fig.add_trace(go.Scatter(x=xgb_pred.index, y=xgb_pred, mode="lines", name="XGBoost Forecast", line=dict(color="cyan", width=2)))

    # TF LSTM Forecast
    if tf_lstm_pred is not None:
        fig.add_trace(go.Scatter(x=tf_lstm_pred.index, y=tf_lstm_pred, mode="lines", name="TF LSTM Forecast", line=dict(color="orange", width=2)))

    # Thresholds
    fig.add_hline(y=2.86*1000, line_dash="dash", line_color="red", annotation_text="Flood Threshold (2.86) in April 1983", annotation_position="top left")
    fig.add_hline(y=2.77*1000, line_dash="dash", line_color="cyan", annotation_text="Flood Threshold (2.77) in February 1990", annotation_position="bottom left")
    fig.add_hline(y=2.58*1000, line_dash="dash", line_color="green", annotation_text="Flood Threshold (2.58) in January 2018", annotation_position="bottom left")

    fig.update_layout(
        title=f"All Models Comparison - {TITLE_POSTFIX_NAME} - Water ({STATION_MEASURE})",
        title_x=0.5,
        title_font=dict(size=24, color="black", family="Arial", weight="bold"),
        xaxis_title="Datetime",
        yaxis_title=f"Water ({STATION_MEASURE})",
        legend_title="Legend",
        width=1400, height=800,
        xaxis=dict(
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
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig


def plot_prophet_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                               prophet_forecast: pd.DataFrame, 
                               TITLE_POSTFIX_NAME: str, STATION_MEASURE: str) -> go.Figure:
    """Generate Plotly graphic only for Prophet Forecast against actual Train/Test sets."""
    # XGBoost Forecast
    xgb_pred = None
    # TF LSTM Forecast
    tf_lstm_pred = None

    fig = plot_all_models_comparison(train_df, test_df, 
                               prophet_forecast, xgb_pred, 
                               tf_lstm_pred, 
                               TITLE_POSTFIX_NAME, STATION_MEASURE)
    return fig


def plot_xgboost_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                               xgb_pred: pd.Series, 
                               TITLE_POSTFIX_NAME: str, STATION_MEASURE: str) -> go.Figure:
    """Generate Plotly graphic only for XGBoost Forecast against actual Train/Test sets."""
    # Prophet Forecast
    prophet_forecast = None
    # TF LSTM Forecast
    tf_lstm_pred = None

    fig = plot_all_models_comparison(train_df, test_df, 
                               prophet_forecast, xgb_pred, 
                               tf_lstm_pred, 
                               TITLE_POSTFIX_NAME, STATION_MEASURE)
    return fig


def plot_tensorflow_lstm_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                               tf_lstm_pred: pd.Series, 
                               TITLE_POSTFIX_NAME: str, STATION_MEASURE: str) -> go.Figure:
    """Generate Plotly graphic only for TF LSTM Forecast against actual Train/Test sets."""
    # Prophet Forecast
    prophet_forecast = None
    # XGBoost Forecast
    xgb_pred = None

    fig = plot_all_models_comparison(train_df, test_df, 
                               prophet_forecast, xgb_pred, 
                               tf_lstm_pred, 
                               TITLE_POSTFIX_NAME, STATION_MEASURE)
    return fig