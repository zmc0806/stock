import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go

st.title('stock data')

ticker_symbol = st.text_input("stock", "AAPL")

start_date = st.date_input("choose start data", pd.to_datetime('2024-01-01'))
end_date = st.date_input("choose end data", pd.to_datetime('today'))

changepoint_prior_scale = st.slider("Changepoint Prior Scale (default 0.05)", 0.001, 0.5, 0.05)
seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
seasonality_prior_scale = st.slider("Seasonality Prior Scale (default 10.0)", 1.0, 50.0, 10.0)

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

        st.write(f"{ticker_symbol} future prediction")

        df_prophet = ticker_df.reset_index()[['Date', 'Close']]
        df_prophet.columns = ['ds', 'y']  
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)  

        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale
        )
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', name='Lower Bound'))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', fill='tonexty', name='Upper Bound'))

        fig_forecast.update_layout(title=f"{ticker_symbol} Future Price Prediction", xaxis_title="Date", yaxis_title="Predicted Price")
        st.plotly_chart(fig_forecast)        
else:
    st.info("Please enter the ticker symbol and click the Get Data button")
