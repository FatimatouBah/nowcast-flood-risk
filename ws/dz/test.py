# -*- coding=utf-8 -*-
import pandas as pd
import numpy

from prophet import Prophet

import argparse

import mlflow
# import mlflow.sklearn
# from mlflow.models import infer_signature
# # from mlflow import log_metric, log_param, log_artifacts

from sklearn.metrics import  (r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error)

from plotly import express as px
from plotly import graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

import os
from datetime import datetime

# Load Data
site_name = "Kogenheim"
code_site ="A2360030"
code_station = "A236003001"
data_df = pd.read_csv(".\\data\\vars\\hubeau_obs_elab_Kogenheim_A236003001_HIXnJ_20070101_20260602.csv")


# get the current working directory
WorkDir = os.path.join(os.getcwd(), "local")
# set current as working directory
os.chdir(WorkDir)
print(f"Current working directory: {os.getcwd()}")

Current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Current time: {Current_time}")

# Create artifact directory (output file) to save the results of the model training and evaluation
local_artifact_dir = os.path.join(WorkDir, f"artifact_outputs_{Current_time}")
if not os.path.exists(local_artifact_dir):
    os.makedirs(local_artifact_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MLflow + scikit-learn training script")
    # parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of test set")
    # parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--seasonality", type=bool, default=False, help="Include seasonality components (True or False)")
    return parser.parse_args()


# run script with arguments example: python test.py --test-size 0.25 --random-state 123

def main() -> None:
    args = parse_args()
    # =======================================================================
    # I) Exploratory Data Analysis (EDA) + Time Series Visualization
    # =======================================================================

    print(type(data_df))
    print(data_df.columns)
    print(data_df.head())
    
    # Define the column names for the datetime and target variable in the dataset
    DATETIME_COLUMN_NAME = "date_obs_elab"
    TARGET_COLUMN_NAME = "resultat_obs_elab"
    TARGET_VARIABLE_NOTATION = "HIXnJ"
    
    TITLE_POSTFIX_NAME = f"{site_name} - {code_site} - {code_station} : {TARGET_VARIABLE_NOTATION}"
    FILE_POSTFIX_NAME = TITLE_POSTFIX_NAME.replace(' ', '').replace(':', '_').replace('-', '_').replace('(', '').replace(')', '')
    
    # I.1. Select the relevant columns and rename them to 'ds' and 'y'
    prophet_df = data_df[[DATETIME_COLUMN_NAME, TARGET_COLUMN_NAME]].copy()
    prophet_df = prophet_df.rename(columns={DATETIME_COLUMN_NAME: 'ds', 
                                            TARGET_COLUMN_NAME: 'y'
                                            })

    # I.2. Ensure the 'ds' column is a proper datetime type.
    # Note: Prophet often prefers timezone-naive datetimes, so we remove the timezone if it exists.
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)

    # I.3. Ensure 'y' is numeric
    # prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')

    # # Optional: drop any rows with NaN values in 'y' as they can cause issues
    # prophet_df = prophet_df.dropna(subset=['y'])

    # I.4. Time Series With Range Slider
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name=f'Water Height ({TARGET_VARIABLE_NOTATION})'))
    # fig1.add_trace(go.Scatter(gold_ts, x="Date", y="Close", mode='lines', name='Close'))
    # fig1.add_trace(go.Scatter(gold_ts, x="Date", y=gold_ts.columns, mode='lines', name='Close'))

    # Customize the current trace
    fig1.update_traces(line_color='blue', name=f'Water Height ({TARGET_VARIABLE_NOTATION})', showlegend=True)

    # add horizontal line for flood threshold (example: 2.86) in April 1983
    fig1.add_hline(y=2.86*1000, line_dash="dash", line_color="red", annotation_text="Flood Threshold (2.86) in April 1983", annotation_position="top left")
    # add horizontal line for flood threshold (example: 2.77) in February 1990
    fig1.add_hline(y=2.77*1000, line_dash="dash", line_color="cyan", annotation_text="Flood Threshold (2.77) in February 1990", annotation_position="bottom left")
    # add horizontal line for flood threshold (example: 2.58) in January 2018
    fig1.add_hline(y=2.58*1000, line_dash="dash", line_color="green", annotation_text="Flood Threshold (2.58) in January 2018", annotation_position="bottom left")

    fig1.update_layout(
        title=f'{TITLE_POSTFIX_NAME} : Water Height Over Time',
        title_x=0.5,
        title_font=dict(size=24, color='black', family='Arial', weight='bold'),
        xaxis_title='Datetime', 
        yaxis_title=f'Water Height ({TARGET_VARIABLE_NOTATION})',
        legend_title='Legend',
    #   showlegend=True,
    #   legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='black', borderwidth=1),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    # dict(count=1, label="1m", step="month", stepmode="backward"),
                    # dict(count=6, label="6m", step="month", stepmode="backward"),
                    # dict(count=1, label="YTD", step="year", stepmode="todate"),
                    # dict(count=1, label="1y", step="year", stepmode="backward"),
                    
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
    # plot and save the figure as an HTML and png file
    fig1.write_html(os.path.join(local_artifact_dir, f"01_time_series_plot_{FILE_POSTFIX_NAME}.html"))
    fig1.write_image(os.path.join(local_artifact_dir, f"01_time_series_plot_{FILE_POSTFIX_NAME}.png"))
    # fig1.show()


    # =======================================================================
    # II) Train/Test Split of the data for time series forecasting with Prophet
    # =======================================================================

    split_date = '2024-01-01'
    # mask_selection = data_df.index < split_date
    # train_fbp = data_df[mask_selection]
    # test_fbp = data_df[~mask_selection]

    train_fbp = prophet_df.loc[prophet_df['ds'] <= split_date].copy()
    test_fbp = prophet_df.loc[prophet_df['ds'] > split_date].copy()
    # save the train and test sets as csv files
    train_fbp.to_csv(os.path.join(local_artifact_dir, f"train_fbp_{FILE_POSTFIX_NAME}.csv"), index=False)
    test_fbp.to_csv(os.path.join(local_artifact_dir, f"test_fbp_{FILE_POSTFIX_NAME}.csv"), index=False)
    # print(train_fbp)

    # Visualize the train/test split using plotly
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train_fbp['ds'], y=train_fbp['y'], mode='lines', name='Train Set'))
    fig2.add_trace(go.Scatter(x=test_fbp['ds'], y=test_fbp['y'], mode='lines', name='Test Set'))
    fig2.add_vline(x=split_date, line_dash='dash', line_color='black')
    # fig2.add_vline(x=split_date, line_dash="dash", line_color="red", annotation_text="Train-Test split line", annotation_position="top left")

    # Customize the layout
    # fig2.update_layout(title='Data Train/Test Split', xaxis_title='Date', yaxis_title='Water Height', width=1200, height=600)
    fig2.update_layout(title=f'Data Train/Test Split - Water Height Over Time - {TITLE_POSTFIX_NAME}',
                    title_x=0.5,
                    title_font=dict(size=24, color='black', family='Arial', weight='bold'),
                    xaxis_title='Datetime', 
                    yaxis_title='Water Height',
                    legend_title='Legend',
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='black', borderwidth=1),
                    # Size of the figure
                    width=1200, height=600
    )

    # plot and save the figure as an HTML and png file
    fig2.write_html(os.path.join(local_artifact_dir, f"02_train_test_split_plot_{FILE_POSTFIX_NAME}.html"))
    fig2.write_image(os.path.join(local_artifact_dir, f"02_train_test_split_plot_{FILE_POSTFIX_NAME}.png"))
    # fig2.show()


    # =======================================================================
    # III) Initialize and train the Prophet model
    # =======================================================================

    # Initialize the Prophet model with or without seasonality components (daily, weekly, yearly) based on the seasonality variable
    seasonality = args.seasonality
    if seasonality:
        prophet_model = Prophet()
    else:
        prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    
    # Fit the Prophet model on the training data
    prophet_model.fit(train_fbp)


    # =======================================================================
    # IV) Predict on test set with model
    # =======================================================================

    test_fbp_prophet = test_fbp.reset_index().rename(columns={DATETIME_COLUMN_NAME:'ds'})
    test_fcst = prophet_model.predict(test_fbp_prophet[["ds"]])
    # save the forecasted results as a csv file
    test_fcst.to_csv(os.path.join(local_artifact_dir, f"test_forecast_{FILE_POSTFIX_NAME}.csv"), index=False)
    # # print the forecasted results
    # print(test_fcst)

    # Plot the components of the model
    fig3 = prophet_model.plot_components(test_fcst)
    # save the figure as an HTML and png file
    fig3.savefig(os.path.join(local_artifact_dir, f"03_prophet_components_plot_{FILE_POSTFIX_NAME}.png"))
    # fig3.show()

    # Plot the components of the model using plotly
    fig4 = make_subplots(rows=4, cols=1, subplot_titles=['Trend', 'Weekly Seasonality', 'Yearly Seasonality', 'Daily Seasonality'])
    fig4.add_trace(go.Scatter(x=test_fcst['ds'], y=test_fcst['trend'], mode='lines', name='Trend'), row=1, col=1)
    fig4.add_trace(go.Scatter(x=test_fcst['ds'], y=test_fcst['weekly'], mode='lines', name='Weekly Seasonality'), row=2, col=1)
    fig4.add_trace(go.Scatter(x=test_fcst['ds'], y=test_fcst['yearly'], mode='lines', name='Yearly Seasonality'), row=3, col=1)
    fig4.add_trace(go.Scatter(x=test_fcst['ds'], y=test_fcst['daily'], mode='lines', name='Daily Seasonality'), row=4, col=1)
    fig4.update_layout(title=f'Prophet Model Components - Water Height Over Time - {TITLE_POSTFIX_NAME}',
                    title_x=0.5,
                        title_font=dict(size=24, color='black', family='Arial', weight='bold'),
                        xaxis_title='Datetime', 
                        yaxis_title='Component Value',
                        legend_title='Legend',
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='black', borderwidth=1),
                        height=900, width=1200
    )
    # save the figure as an HTML and png file
    fig4.write_html(os.path.join(local_artifact_dir, f"04_prophet_components_plot_{FILE_POSTFIX_NAME}.html"))
    fig4.write_image(os.path.join(local_artifact_dir, f"04_prophet_components_plot_{FILE_POSTFIX_NAME}.png"))
    # fig4.show()


    # =======================================================================
    # V) Create a future dataframe and forecast
    # =======================================================================

    # # Forecasting the next 24 hours (freq='H') -> 1 day only
    # future = prophet_model.make_future_dataframe(periods=24, freq='H')
    # Forecasting the next 30 days (freq='D') -> 1 month only
    future = prophet_model.make_future_dataframe(periods=30, freq='D') 
    # # Forecasting the next 365 days (freq='D') -> 1 year only
    # future = prophet_model.make_future_dataframe(periods=365, freq='D')
    # # Forecasting the next 30 days (freq='D') with daily seasonality only
    # future = prophet_model.make_future_dataframe(periods=30, freq='D', include_history=True)
    # # Forecasting the next 1 year (freq='D') with daily seasonality only
    # future = prophet_model.make_future_dataframe(periods=365, freq='D', include_history=True)

    # Generate the forecast for the future dataframe for the next 24 hours or 30 days
    forecast = prophet_model.predict(future)

    print("Future dataframe created:")
    print(type(future))
    print("The first 5 rows of the future dataframe:")
    print(future.head())
    print("The last 5 rows of the future dataframe:")
    print(future.tail())

    # View the forecasted results
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    print("\nForecast dataframe created:")
    print(type(forecast))
    print("The first 5 rows of the forecast dataframe:")
    print(forecast.head())
    print("The last 5 rows of the forecast dataframe:")
    print(forecast.tail())


    # Plot the forecasted results
    fig5 = prophet_model.plot(forecast)
    fig5.savefig(os.path.join(local_artifact_dir, f"05_prophet_forecast_plot_{FILE_POSTFIX_NAME}.png"))
    # fig5.show()

    # Plot the forecasted results using plotly
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Water Height (yhat)'))
    fig6.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Confidence Interval (yhat_lower)', line=dict(dash='dash', color='red')))
    fig6.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Confidence Interval (yhat_upper)', line=dict(dash='dash', color='green')))
    fig6.update_layout(title=f'Prophet Forecast - Water Height Over Time - {TITLE_POSTFIX_NAME}',
                    title_x=0.5,
                        title_font=dict(size=24, color='black', family='Arial', weight='bold'),
                        xaxis_title='Datetime', 
                        yaxis_title='Water Height',
                        legend_title='Legend',
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='black', borderwidth=1),
                        height=900, width=1200
    )
    # save the figure as an HTML and png file
    fig6.write_html(os.path.join(local_artifact_dir, f"06_prophet_forecast_plot_{FILE_POSTFIX_NAME}.html"))
    fig6.write_image(os.path.join(local_artifact_dir, f"06_prophet_forecast_plot_{FILE_POSTFIX_NAME}.png"))
    # fig6.show()


    # Plot the forecast with confidence intervals and actual history using plotly with interactive plot and range slider options
    # We visualize the forecast, confidence intervals, and actual history.
    # Use plotly only to visualize the forecast, confidence intervals, and actual history with interactive plot and range slider options
    fig7 = go.Figure()

    # Add the actual training data
    fig7.add_trace(go.Scatter(x=train_fbp['ds'], y=train_fbp['y'], mode='lines+markers', name='Actual (Training)', line=dict(color='blue')))

    # Add the actual test data
    fig7.add_trace(go.Scatter(x=test_fbp['ds'], y=test_fbp['y'], mode='lines+markers', name='Actual (Test)', line=dict(color='green')))

    # Add the forecast line
    fig7.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red')))

    # Add confidence interval upper bound (invisible line to define fill boundary)
    fig7.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', 
                            line_color='rgba(0,0,0,0)', showlegend=False))

    # Add confidence interval lower bound with fill to upper bound
    fig7.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', 
                            line_color='rgba(0,0,0,0)', name='Confidence Interval', 
                            fillcolor='rgba(255,0,0,0.2)'))

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

    # save the figure as an HTML and png file
    fig7.write_html(os.path.join(local_artifact_dir, f"07_prophet_forecast_with_history_plot_{FILE_POSTFIX_NAME}.html"))
    fig7.write_image(os.path.join(local_artifact_dir, f"07_prophet_forecast_with_history_plot_{FILE_POSTFIX_NAME}.png"))
    # fig7.show()


    # =======================================================================
    # VI) Compute the metrics (MAPE, R^2, RMSE, MAE) to evaluate the forecast accuracy
    # =======================================================================

    # We now measure how accurate the forecast is. If there’s no overlap in dates, it warns you instead of crashing.

    #VI.1. Align predictions with actuals
    # Join actual water height with model predictions on matching dates to compare them directly.

    # Join actual closing prices with model predictions on matching dates to compare them directly.
    # We merge the test dataframe (which contains the actual closing prices for the test period) with the forecast dataframe (which contains the predicted values for the same period) on the "ds" column, 
    # which represents the dates.
    # The merge is done using an inner join, which means that only the rows with matching dates in both dataframes will be included in the resulting merged dataframe.
    merged = pd.merge(test_fbp, forecast[["ds", "yhat"]], on="ds", how="inner")

    print("\nMerged dataframe created:")
    print(type(merged))
    print("\nThe first 5 rows of the merged dataframe:")
    print(merged.head())
    print("\nThe last 5 rows of the merged dataframe:")
    print(merged.tail())

    # VI.2. Compute (r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error)
    # We now measure how accurate the forecast is. If there’s no overlap in dates, it warns you instead of crashing.

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
        
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}")
        print(f"R^2 score: {r2:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        
        
        with open(os.path.join(local_artifact_dir, f"train_{FILE_POSTFIX_NAME}_{Current_time}.txt"), "w") as f:
            f.write(f"Current date and time: {Current_time}\n")
            f.write(f"{TITLE_POSTFIX_NAME}\n")
            f.write("Model Type: Linear Regression\n")
            f.write("\nParameters:\n")
            # f.write(f"\t-Test Size: {args.test_size}\n")
            # f.write(f"\t-Random State: {args.random_state}\n")
            f.write(f"\t-Seasonality: {args.seasonality}\n")
            f.write("\nThis is an artifact file containing the evaluation metrics of the model:\n")
            f.write(f"\t-MAPE: {mape}\n")
            f.write(f"\t-R^2 Score: {r2}\n")
            f.write(f"\t-MSE: {mse:.4f}\n")
            f.write(f"\t-RMSE: {rmse:.4f}\n")
            f.write(f"\t-MAE: {mae:.4f}\n")
            

    # VI. Plot seasonal components
    # It's helpful to show how trend, daily, and yearly seasonality patterns contribute to the forecast.

    fig8 = prophet_model.plot_components(forecast)
    fig8.savefig(os.path.join(local_artifact_dir, f"08_prophet_seasonal_components_plot_{FILE_POSTFIX_NAME}.png"))
    # fig8.show()

    # Use plotly to plot seasonal components
    fig9 = make_subplots(rows=4, cols=1, subplot_titles=['Trend', 'Weekly Seasonality', 'Yearly Seasonality', 'Daily Seasonality'])
    fig9.add_trace(go.Scatter(x=test_fcst['ds'], y=test_fcst['trend'], mode='lines', name='Trend'), row=1, col=1)
    fig9.add_trace(go.Scatter(x=test_fcst['ds'], y=test_fcst['weekly'], mode='lines', name='Weekly Seasonality'), row=2, col=1)
    fig9.add_trace(go.Scatter(x=test_fcst['ds'], y=test_fcst['yearly'], mode='lines', name='Yearly Seasonality'), row=3, col=1)
    fig9.add_trace(go.Scatter(x=test_fcst['ds'], y=test_fcst['daily'], mode='lines', name='Daily Seasonality'), row=4, col=1)
    fig9.update_layout(title=f'Prophet Seasonal Components - Water Height Over Time - {TITLE_POSTFIX_NAME}',
                    title_x=0.5,
                        title_font=dict(size=24, color='black', family='Arial', weight='bold'),
                        xaxis_title='Datetime', 
                        yaxis_title='Component Value',
                        legend_title='Legend',
                    #   showlegend=True,
                    #   legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='black', borderwidth=1),
                        height=900, width=1200
    )
    # save the figure as an HTML and png file
    fig9.write_html(os.path.join(local_artifact_dir, f"09_prophet_seasonal_components_plot_{FILE_POSTFIX_NAME}.html"))
    fig9.write_image(os.path.join(local_artifact_dir, f"09_prophet_seasonal_components_plot_{FILE_POSTFIX_NAME}.png"))
    # fig9.show()


if __name__ == "__main__":
    print("Starting the test script...")
    # main()