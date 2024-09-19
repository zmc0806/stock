import streamlit as st
import yfinance as yf
import pandas as pd


st.title('stock data')

ticker_symbol = st.text_input("stock", "AAPL")

start_date = st.date_input("choose start data", pd.to_datetime('2024-01-01'))
end_date = st.date_input("choose end data", pd.to_datetime('today'))

if st.button('get data'):
    ticker_data = yf.Ticker(ticker_symbol)
    ticker_df = ticker_data.history(period='1d', start=start_date, end=end_date)
    st.write(f"{ticker_symbol} stock data")
    st.write(ticker_df)



