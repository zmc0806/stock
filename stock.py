import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title('Stock Data With Linear Regression Prediction')

ticker_symbol = st.text_input("stock", "AAPL")

start_date = st.date_input("choose start data", pd.to_datetime('2024-01-01'))
end_date = st.date_input("choose end data", pd.to_datetime('today'))

if st.button('get data'):
    ticker_data = yf.Ticker(ticker_symbol)
    ticker_df = ticker_data.history(period='1d', start=start_date, end=end_date)
  
    if ticker_df.empty:
        st.error(f"can not {ticker_symbol} stock data please check")
    else:
        st.write(f"{ticker_symbol} stock data")
        st.write(ticker_df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_df.index, y=ticker_df['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f"{ticker_symbol} Close Price Time Series", xaxis_title="Date", yaxis_title="Close Price")
        st.plotly_chart(fig)

        st.write(f"Linear Regression model for {ticker_symbol}")


        # df_prophet = ticker_df.reset_index()[['Date', 'Close']]
        # df_prophet.columns = ['ds', 'y']  
        # df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)  

        # model = Prophet()
        # model.fit(df_prophet)

        # future = model.make_future_dataframe(periods=365)
        # forecast = model.predict(future)

        # st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
        # fig_forecast = go.Figure()
        # fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
        # fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', name='Lower Bound'))
        # fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', fill='tonexty', name='Upper Bound'))

        # fig_forecast.update_layout(title=f"{ticker_symbol} Future Price Prediction", xaxis_title="Date", yaxis_title="Predicted Price")
        # st.plotly_chart(fig_forecast)
        ticker_df['Time'] = np.arange(len(ticker_df))
        X = ticker_df['Time'].values
        y = ticker_df['Close'].values

        n = len(X)
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        m = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
        b = y_mean - m * X_mean

        future_time = np.arange(len(ticker_df), len(ticker_df) + 365)
        predicted_prices = m * future_time + b

        future_dates = pd.date_range('2024-01-01', end=end_date)
        ticker_future_df = ticker_data.history(period='1d', start='2024-01-01', end=end_date)

        predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predicted_prices[:len(future_dates)]})

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=ticker_future_df.index, y=ticker_future_df['Close'], mode='lines', name='Actual Close Price'))
        fig_compare.add_trace(go.Scatter(x=predicted_df['Date'], y=predicted_df['Predicted_Close'], mode='lines', name='Manual Linear Regression Predicted'))

        fig_compare.update_layout(title=f"{ticker_symbol} Actual vs Predicted (Linear Regression)", xaxis_title="Date", yaxis_title="Close Price")
        st.plotly_chart(fig_compare)

        actual_prices = ticker_future_df['Close'].values
        predicted_prices = predicted_df['Predicted_Close'].values[:len(actual_prices)]

        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_prices, predicted_prices)
        r2 = r2_score(actual_prices, predicted_prices)

        st.write(f"Prediction Accuracy for {ticker_symbol}:")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")        
else:
    st.info("Please enter the ticker symbol and click the Get Data button")
