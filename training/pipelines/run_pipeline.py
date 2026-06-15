# -*- coding=utf-8 -*-
# Import necessary system and path manipulation modules
import sys
from pathlib import Path
import os
import time
from datetime import datetime, date

# Import MLflow for experiment tracking and model logging
import mlflow

# Add parent directory to path to allow importing config modules
# This ensures that we can import from the 'config' and 'models' directories correctly
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import utilities for fetching data from the API, cleaning the time series, and splitting the data
from config.data_utils import fetch_obs_elab, prepare_series, split_train_test, SITES
# Import utilities for argument parsing, MLflow configuration, and plotting
from config.model_utils import parse_args, setup_mlflow, plot_all_models_comparison

# Get current directory of this script file
PROJECT_DIR = Path(__file__).resolve().parent
# Create directory Plot if not exist
Plot_DIR = PROJECT_DIR / "Plot"
Plot_DIR.mkdir(exist_ok=True)

# Import the main training functions for each forecasting model
from models.prophet.train_prophet import main_train_prophet
from models.xgboost.train_xgboost import main_train_xgboost
from models.lstm.train_lstm_tensorflow import main_train_lstm


def main():
    """
    Main orchestration function to run the entire data and training pipeline.
    It fetches data, splits it, trains four different models, and evaluates them.
    """
    # Parse command-line arguments to configure the pipeline execution
    args = parse_args()
    
    # Extract arguments into local variables for easier access
    site_name = args.site_name
    code_site = args.code_site
    code_station = args.code_station
    site_measure = args.site_measure
    start_date = args.start_date
    end_date = str(args.end_date)
    
    # Extract all valid site codes from the configuration dictionary
    site_codes = [SITES[name]["code"] for name in SITES.keys()]
    
    # Determine the default site code based on the provided site_name, or fallback to Kogenheim
    default_code_site = SITES[site_name]["code"] if site_name in SITES else SITES["Kogenheim"]["code"]
    
    # Fallback to the default site code if the provided one is invalid
    if code_site not in site_codes:
        code_site = default_code_site
        
    # Get the specific configuration block for the selected site
    site_config = SITES.get(site_name, SITES["Kogenheim"])
    
    # Extract valid station codes for the chosen site
    station_codes = [item["code"] for item in site_config["stations"]]
    
    # Fallback to the first available station code if the provided one is invalid
    if code_station not in station_codes:
        code_station = station_codes[0]

    # Log the parameters being used to fetch the data
    print(f"Loading data for site: {site_name}, station: {code_station}, from {start_date} to {end_date}...")
    
    # Step 1: Fetch Data from the Hub'Eau API
    raw_df = fetch_obs_elab(code_station, start_date, end_date, site_measure)
    
    # Check if the fetched dataframe is empty and exit gracefully if it is
    if raw_df.empty:
        print("No data fetched from API. Exiting.")
        return
        
    # Step 2: Prepare Series for modeling (renaming columns to ds/y, dropping nulls, handling dates)
    clean_df = prepare_series(raw_df)
    
    # Check if the cleaning process removed all data
    if clean_df.empty:
        print("Data could not be prepared. Exiting.")
        return
        
    # Step 3: Train/Test Split
    # We split the chronological time series by an 80/20 ratio for training and testing
    train_df, test_df, SPLIT_DATE = split_train_test(clean_df, train_ratio=0.8, split_by_ratio=True)
    
    # Verify that the split returned valid datasets
    if train_df is None or test_df is None:
        print("Not enough data to split. Exiting.")
        return
        
    # Print a summary of the data size and split cutoff
    print(f"Data ready. Train size: {len(train_df)}, Test size: {len(test_df)}, Split date: {SPLIT_DATE}")
    
    # Step 4: Setup MLFlow Environment variables (Tracking URI, Registry URI)
    setup_mlflow()
    
    # Define a common title postfix for the plots
    TITLE_POSTFIX_NAME = f"{site_name} - {code_station}"

    
    start_time = time.time()
    print("\n" + "="*20)
    print("Starting Prophet model training...")
    print("Time start (seconds):", start_time)
    print("="*20+"\n")
    # =======================================================================
    # 1. Run Prophet Baseline Model
    # =======================================================================
    # Prophet provides a baseline forecast and decomposition components
    prophet_model, prophet_forecast, prophet_metrics = main_train_prophet(
        train_df, test_df, site_name, code_station, code_site, site_measure, SPLIT_DATE, TITLE_POSTFIX_NAME,
        seasonality=args.seasonality, 
        changepoint_prior_scale=args.changepoint_prior_scale, 
        seasonality_prior_scale=args.seasonality_prior_scale
    )

    print("\n" + "="*20)
    print("Prophet model training completed.")
    print("Total time taken (seconds):", time.time() - start_time)
    print("="*20+"\n") 


    start_time = time.time()
    print("\n" + "="*20)
    print("Starting XGBoost model training...")
    print("Time start (seconds):", start_time)
    print("="*20+"\n")
    # =======================================================================
    # 2. Run XGBoost Model
    # =======================================================================
    # XGBoost trains on extracted time-features (hour, day, month, etc.)
    xgb_forecast = main_train_xgboost(
        train_df, test_df, site_name, code_station, code_site, site_measure, SPLIT_DATE, prophet_forecast, args.random_state, TITLE_POSTFIX_NAME
    )
    
    print("\n" + "="*20)
    print("XGBoost model training completed.")
    print("Total time taken (seconds):", time.time() - start_time)
    print("="*20+"\n") 

    
    start_time = time.time()
    print("\n" + "="*20)
    print("Starting TensorFlow LSTM model training...")
    print("Time start (seconds):", start_time)
    print("="*20+"\n")    
    # =======================================================================
    # 3. Run TensorFlow LSTM Model
    # =======================================================================
    # LSTM learns sequential dependencies using look-back windows
    tf_lstm_forecast = main_train_lstm(
        train_df, test_df, site_name, code_station, code_site, site_measure, SPLIT_DATE, prophet_forecast, TITLE_POSTFIX_NAME
    )

    print("\n" + "="*20)
    print("TensorFlow LSTM model training completed.")
    print("Total time taken (seconds):", time.time() - start_time)
    print("="*20+"\n") 
    
    # =======================================================================
    # 4. Master Comparison & Final MLflow logging
    # =======================================================================
    # Define an overarching MLflow experiment for the full pipeline run
    MLFLOW_EXPERIMENT_NAME = "final_project_forecasting_hubeau/master_pipeline"
    
    # Check if the experiment exists, otherwise create it
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
        
    # Set the active experiment to the master pipeline experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Start the MLflow run, naming it specifically to the station being processed
    with mlflow.start_run(run_name=f"pipeline_station_{code_station}") as run:
        
        # Log the site metadata to the MLflow run
        mlflow.log_param("site_name", site_name)
        mlflow.log_param("code_station", code_station)
        
        # Announce the generation of the unified comparison plot
        print("Generating unified comparison plot...")
        
        # Create a single Plotly figure containing train, test, and all 4 model forecasts
        fig = plot_all_models_comparison(
            train_df, test_df,
            prophet_forecast, xgb_forecast, tf_lstm_forecast,
            TITLE_POSTFIX_NAME, site_measure
        )
        
        # Log the generated Plotly figure natively to MLflow as a PNG artifact
        fig.write_html(str(PROJECT_DIR / f"master_comparison_plot_{code_station}.html"))
        mlflow.log_artifact(str(PROJECT_DIR / f"master_comparison_plot_{code_station}.html"), artifact_path="Plots")
        
        # Temporarily save the cleaned dataset to CSV for artifact logging
        clean_df.to_csv(str(PROJECT_DIR / f"clean_data_{code_station}.csv"), index=False)
        
        # Log the dataset artifact to MLflow to ensure complete reproducibility
        mlflow.log_artifact(str(PROJECT_DIR / f"clean_data_{code_station}.csv"), artifact_path="Data")
        
        # Remove the local temporary CSV file to keep the disk clean
        os.remove(str(PROJECT_DIR / f"clean_data_{code_station}.csv"))
        
        # Print a final success message indicating the orchestration has completed
        print("Pipeline execution completed successfully. All artifacts logged to MLflow.")

# Entry point for the script execution
if __name__ == "__main__":
    main()