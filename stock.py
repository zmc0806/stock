import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title('Stock Data with Linear Regression Prediction (Manual)')

ticker_symbol = st.text_input("Stock", "AAPL")

train_start_date = st.date_input("choose start data", pd.to_datetime('2020-01-01'))
train_end_date = st.date_input("choose start data", pd.to_datetime('2023-12-31'))
predict_start_date = st.date_input("choose start data", pd.to_datetime('2024-01-01'))
predict_end_date = st.date_input("choose end data", pd.to_datetime('today'))


if st.button('Get Data'):
    ticker_data = yf.Ticker(ticker_symbol)
    train_df = ticker_data.history(period='1d', start=train_start_date, end=train_end_date)


    if train_df.empty:
        st.error(f"Cannot retrieve {ticker_symbol} stock data, please check the input.")
    else:
        st.write(f"{ticker_symbol} stock data (2020-2023)")
        st.write(train_df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f"{ticker_symbol} Close Price Time Series (2020-2023)", xaxis_title="Date", yaxis_title="Close Price")
        st.plotly_chart(fig)

        st.write(f"Linear Regression model for {ticker_symbol} (2020-2023)")

        train_df['Time'] = np.arange(len(train_df))  # Time as the feature
        X_train = train_df['Time'].values
        y_train = train_df['Close'].values

        X_mean = np.mean(X_train)
        y_mean = np.mean(y_train)
        m = np.sum((X_train - X_mean) * (y_train - y_mean)) / np.sum((X_train - X_mean)**2)
        b = y_mean - m * X_mean

        ticker_future_df = ticker_data.history(period='1d', start=predict_start_date, end=predict_end_date)
        future_dates = ticker_future_df.index

        future_time = np.arange(len(train_df), len(train_df) + len(future_dates))
        predicted_prices = m * future_time + b

        predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predicted_prices})

        if not ticker_future_df.empty:
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(x=ticker_future_df.index, y=ticker_future_df['Close'], mode='lines', name='Actual Close Price'))
            fig_compare.add_trace(go.Scatter(x=predicted_df['Date'], y=predicted_df['Predicted_Close'], mode='lines', name='Predicted Close Price'))
            fig_compare.update_layout(title=f"{ticker_symbol} Actual vs Predicted (2024)", xaxis_title="Date", yaxis_title="Close Price")
            st.plotly_chart(fig_compare)

            actual_prices = ticker_future_df['Close'].values
            predicted_prices = predicted_df['Predicted_Close'].values[:len(actual_prices)]

            mse = mean_squared_error(actual_prices, predicted_prices)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_prices, predicted_prices)
            r2 = r2_score(actual_prices, predicted_prices)

            st.write(f"Prediction Accuracy for {ticker_symbol} (2024):")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            st.write(f"R-squared (R²): {r2:.2f}")
        else:
            st.warning(f"No actual data available for {ticker_symbol} in 2024 for comparison.")

else:
    st.info("Please enter the ticker symbol and click the Get Data button.")
