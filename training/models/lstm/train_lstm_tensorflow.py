# -*- coding=utf-8 -*-
# Import necessary modules for path manipulation and standard OS operations
from pathlib import Path
import os
import time

# Import pandas and numpy for data manipulation and array operations
import pandas as pd
import numpy as np

# Import MLflow for experiment tracking, model logging, and parameter tracking
import mlflow
import mlflow.tensorflow

# Import scikit-learn utilities for data scaling and metric calculation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Import TensorFlow and Keras components for building the LSTM neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import the shared MLflow model registration utility
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.model_utils import register_and_alias_model, plot_tensorflow_lstm_forecast

# Get current directory of this script file
CURRENT_DIR = Path(__file__).resolve().parent
# Create directory Plot if not exist
Plot_DIR = CURRENT_DIR / "Plot"
Plot_DIR.mkdir(exist_ok=True)

import warnings
# Filter out non-critical warnings to keep the execution log clean
warnings.filterwarnings("ignore")

# Specify to use GPU or CPU device depending on system availability
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    device_name = '/device:CPU:0'
# Set the visible devices for TensorFlow so it explicitly runs on the chosen hardware
tf.config.set_visible_devices([], 'GPU') if device_name == '/device:CPU:0' else tf.config.set_visible_devices([], 'CPU')

def LSTM_prediction(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict):
    """
    Constructs, trains, and uses an LSTM neural network to generate time series predictions.
    """
    # Extract training parameters from the configuration dictionary
    look_back = params["look_back"]           # Number of previous time steps to use as input variables to predict the next time period
    batch_size = params["batch_size"]         # Number of samples that will be propagated through the network
    epochs = params["epochs"]                 # Number of times the learning algorithm will work through the entire training dataset
    lstm_units_1 = params["lstm_units_1"]     # Dimensionality of the output space for the first LSTM layer
    lstm_units_2 = params["lstm_units_2"]     # Dimensionality of the output space for the second LSTM layer
    dropout_rate = params["dropout_rate"]     # Fraction of the input units to drop during training to prevent overfitting
    
    # Merge the training and testing sets into a single dataframe for sequential processing
    df = pd.concat([train_df, test_df])
    
    # Set the datetime column 'ds' as the index if it exists in the dataframe columns
    df = df.set_index('ds') if 'ds' in df.columns else df
    
    # Isolate only the target variable 'y' to train a Univariate LSTM model
    df = df[['y']] 
    
    # Identify the index where the training data ends and testing data begins
    split_idx = len(train_df)
    
    # Initialize the MinMaxScaler to normalize the data between 0 and 1, which helps neural networks converge faster
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Split the isolated target variable array into training and testing chunks
    train_data = df.iloc[:split_idx].values
    test_data = df.iloc[split_idx:].values
    
    # Fit the scaler on the training data ONLY, and transform both train and test data to avoid data leakage
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    def create_sequences(data, look_back_steps):
        """Helper function to reshape the 1D array into sliding windows (sequences) for the LSTM."""
        X, y = [], []
        # Loop through the data to create windows of size 'look_back_steps' and a corresponding target 'y'
        for i in range(look_back_steps, len(data)):
            X.append(data[i - look_back_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    # Create the sequences for the training dataset
    X_train, y_train = create_sequences(train_scaled, look_back)
    
    # Reshape the input to match the 3D format required by LSTM layers: [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # For the test set, we must include the last 'look_back' elements from the training set so the first test point has historical context
    full_test_input = np.vstack([train_scaled[-look_back:], test_scaled])
    
    # Create the sequences for the testing dataset
    X_test, y_test = create_sequences(full_test_input, look_back)
    
    # Reshape the test sequences into the 3D LSTM format
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Initialize the Sequential model framework from Keras
    model = Sequential([
        # Define the input layer shape
        Input(shape=(X_train.shape[1], 1)),
        # Add the first LSTM layer. return_sequences=True passes the full sequence to the next LSTM layer
        LSTM(lstm_units_1, return_sequences=True),
        # Add a Dropout layer to randomly zero out a fraction of inputs to prevent overfitting
        Dropout(dropout_rate),
        # Add the second LSTM layer. return_sequences=False means it only returns the final output of the sequence
        LSTM(lstm_units_2, return_sequences=False),
        # Add a Dropout layer
        Dropout(dropout_rate),
        # Add a fully connected Dense layer to interpret the features extracted by the LSTM layers
        Dense(16, activation='relu'),
        # Add the final output layer to predict a single continuous value
        Dense(1)
    ])
    
    # Compile the model using the Adam optimizer and Mean Squared Error as the loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Define callbacks to optimize the training process dynamically
    callbacks = [
        # Stop training when the validation loss stops improving for 'patience' epochs, and revert to the best weights
        EarlyStopping(monitor='val_loss', patience=params["early_stopping_patience"], restore_best_weights=True),
        # Reduce the learning rate by a factor of 0.5 when the validation loss plateaus
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=params["learning_patience"], min_lr=1e-6)
    ]
    
    # Fit the compiled model to the training sequences, reserving 10% of the training data as an internal validation set
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_split=0.1, callbacks=callbacks, verbose=params["verbose"], shuffle=False)
    
    # =======================================================================
    # Predict over all data (Train + Test)
    # =======================================================================
    # Concatenate the scaled training and testing arrays vertically
    full_scaled = np.vstack([train_scaled, test_scaled])
    
    # Pad the beginning of the array with zeros matching the look_back length so the model can predict the very first data points
    padded_full = np.vstack([np.zeros((look_back, 1)), full_scaled])
    
    # Create sequences covering the entire chronological dataset
    X_all, _ = create_sequences(padded_full, look_back)
    
    # Reshape the full dataset sequences for the LSTM
    X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1], 1))
    
    # Generate predictions for the entire dataset
    pred_scaled = model.predict(X_all)
    
    # Inverse transform the predictions from the 0-1 scale back to their original physical units (Water Height)
    predictions = scaler.inverse_transform(pred_scaled)
    
    # Return the trained model, the generated predictions mapped to the original datetime index, and the fitted scaler object
    return model, pd.Series(predictions.flatten(), index=df.index), scaler

def main_train_lstm(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                    site_name: str, code_station: str, code_site:str, site_measure:str, 
                    SPLIT_DATE: str, prophet_forecast: pd.DataFrame = None, TITLE_POSTFIX_NAME: str="") -> pd.Series:
    """
    The main execution function for the TensorFlow LSTM model.
    It manages hyperparameter configuration, MLflow tracking, training execution, metric computation, and logging.
    """
    
    # Define a dictionary containing the hyperparameters for the LSTM architecture and the training process
    params = {
        "look_back": 30,                 # Size of the sliding window used to predict the next value
        "test_size": 0.20,               # Proportion of data used for testing (historical reference)
        "epochs": 60,                    # Maximum number of training iterations
        "batch_size": 32,                # Number of sequences processed before updating weights
        "lstm_units_1": 64,              # Neurons in the first LSTM layer
        "lstm_units_2": 32,              # Neurons in the second LSTM layer
        "dropout_rate": 0.20,            # Dropout probability
        "learning_patience": 5,          # Epochs to wait before reducing the learning rate
        "early_stopping_patience": 10,   # Epochs to wait before stopping training entirely
        "verbose": 0,                    # Suppress TensorFlow console output during training
        "random_seed": 42                # Seed value for reproducibility
    }
    
    # Set the global random seed for TensorFlow to ensure deterministic, reproducible results
    tf.random.set_seed(params["random_seed"])
    # Set the global random seed for NumPy to ensure deterministic, reproducible results
    np.random.seed(params["random_seed"])
    
    # Define MLflow Registry variables
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_TRACKING_URI")
    REGISTERED_MODEL_NAME = "flood_forecast_model_lstm_tf"
    MODEL_ALIAS_NAME = "challenge"
    
    # Set the URIs
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)

    # Define a standard MLflow experiment name for TensorFlow LSTM executions
    MLFLOW_EXPERIMENT_NAME = "final_project_forecasting_hubeau/lstm_tensorflow"
    
    # Check if the MLflow experiment exists in the tracking server
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        # Create the experiment if it does not currently exist
        mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
        
    # Set the current active MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Start a new MLflow run with a specific run name tying it to the current station
    with mlflow.start_run(run_name=f"lstm_station_{code_station}") as run:
        
        # Log metadata parameters to the MLflow run
        mlflow.log_param("site_name", site_name)
        mlflow.log_param("code_station", code_station)
        mlflow.log_param("data_split_date", SPLIT_DATE)
        
        # Log the dictionary of custom hyperparameters to the MLflow run
        mlflow.log_params(params)
        
        # Record the start time to monitor training duration
        start_time = time.time()
        print(f"[{start_time}] Training TF LSTM model...")
        
        # Trigger the model training and prediction process
        model, all_predictions, scaler = LSTM_prediction(train_df, test_df, params)
        
        # Extract the actual test set values to evaluate accuracy
        test_actuals = test_df['y'].values
        
        # Isolate the segment of the master prediction series that corresponds chronologically to the test set
        test_preds = all_predictions.iloc[-len(test_df):].values
        
        # Calculate regression metrics by comparing the test set predictions against the actual test set values
        metrics = {
            "std_mape": mean_absolute_percentage_error(test_actuals, test_preds), # Mean Absolute Percentage Error
            "std_r2": r2_score(test_actuals, test_preds),                         # R-squared value
            "std_mse": mean_squared_error(test_actuals, test_preds),              # Mean Squared Error
            "std_rmse": root_mean_squared_error(test_actuals, test_preds),        # Root Mean Squared Error
            "std_mae": mean_absolute_error(test_actuals, test_preds)              # Mean Absolute Error
        }

        # Generate Plotly graphic only for LSTM Forecast against actual Train/Test sets.
        lstm_fig = plot_tensorflow_lstm_forecast(train_df, test_df, all_predictions, TITLE_POSTFIX_NAME, site_measure)
        # Save the figure to an HTML file
        lstm_fig.write_html(str(Plot_DIR / f"02_lstm_forecast_plot_{code_station}.html"))
        
        # Log the figures to MLflow as an HTML artifact for dashboard visualization
        mlflow.log_artifact(str(Plot_DIR))
        
        # Iterate over the calculated metrics and log each one natively to MLflow
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            
        # Print the metrics to the console for real-time review
        print("TF LSTM metrics:", metrics)
        
        # Log the 
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("run_type", "lstm_validation")

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

        model_info = mlflow.tensorflow.log_model(
            model=model, 
            name=f"lstm_model_{code_station}",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=train_df.drop(columns=["y"]).iloc[:5]
        )
        
        # Register the model, set its alias, and transition to Production
        register_and_alias_model(
            model_uri=model_info.model_uri,
            registered_model_name=REGISTERED_MODEL_NAME,
            alias_name=MODEL_ALIAS_NAME,
            tracking_uri=MLFLOW_TRACKING_URI,
            registry_uri=MLFLOW_REGISTRY_URI
        )
        
        # Return the comprehensive forecast series to be used by the pipeline orchestrator
        return all_predictions