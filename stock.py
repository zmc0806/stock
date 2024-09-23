import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet


st.title('stock data')

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

        st.line_chart(ticker_df['Close'])

        st.write(f"{ticker_symbol} future prediction")

        df_prophet = ticker_df.reset_index()[['Date', 'Close']]
        df_prophet.columns = ['ds', 'y']  
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)  

        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
else:
    st.info("Please enter the ticker symbol and click the Get Data button")
