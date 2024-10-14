import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class LinearRegressionModel:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X, y):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        self.m = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
        self.b = y_mean - self.m * X_mean

    def predict(self, X):
        return self.m * X + self.b

class LSTMModel:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, dataset):
        X, y = [], []
        for i in range(len(dataset) - self.look_back):
            X.append(dataset[i:(i + self.look_back), 0])
            y.append(dataset[i + self.look_back, 0])
        return np.array(X), np.array(y)

    def fit(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = self.create_dataset(scaled_data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model.fit(X, y, batch_size=32, epochs=100, verbose=0)

    def predict(self, data):
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        X = []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i:(i + self.look_back), 0])
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)

class TransformerModel:
    def __init__(self, look_back=60, num_features=1):
        self.look_back = look_back
        self.num_features = num_features
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, dataset):
        X, y = [], []
        for i in range(len(dataset) - self.look_back):
            X.append(dataset[i:(i + self.look_back), 0])
            y.append(dataset[i + self.look_back, 0])
        return np.array(X), np.array(y)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = Dense(ff_dim, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.look_back, self.num_features))
        x = self.transformer_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)
        x = self.transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)
        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        outputs = Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def fit(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = self.create_dataset(scaled_data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = self.build_model()
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model.fit(X, y, batch_size=32, epochs=100, verbose=0)

    def predict(self, data):
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        X = []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i:(i + self.look_back), 0])
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)

def fetch_stock_data(ticker_symbol, start_date, end_date):
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(period='1d', start=start_date, end=end_date)
    return df

def plot_stock_data(df, ticker_symbol, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f"{ticker_symbol} {title}", xaxis_title="Date", yaxis_title="Close Price")
    st.plotly_chart(fig)

def plot_comparison(actual_df, predicted_df, ticker_symbol, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df.index, y=actual_df['Close'], mode='lines', name='Actual Close Price'))
    fig.add_trace(go.Scatter(x=predicted_df['Date'], y=predicted_df['Predicted_Close'], mode='lines', name='Predicted Close Price'))
    fig.update_layout(title=f"{ticker_symbol} {title}", xaxis_title="Date", yaxis_title="Close Price")
    st.plotly_chart(fig)

def calculate_metrics(actual_prices, predicted_prices):
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    return mse, rmse, mae, r2

def display_metrics(mse, rmse, mae, r2, ticker_symbol):
    st.write(f"Prediction Accuracy for {ticker_symbol}:")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"R-squared (R²): {r2:.2f}")

def calculate_returns(df):
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    return df

def plot_returns(df, ticker_symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Return'], mode='lines', name='Cumulative Return'))
    fig.update_layout(title=f"{ticker_symbol} Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig)

def main():
    st.title('Advanced Stock Data Analysis and Prediction')
    ticker_symbol = st.text_input("Stock", "AAPL")
    train_start_date = st.date_input("Choose train start date", pd.to_datetime('2020-01-01'))
    train_end_date = st.date_input("Choose train end date", pd.to_datetime('2023-12-31'))
    predict_start_date = st.date_input("Choose prediction start date", pd.to_datetime('2024-01-01'))
    predict_end_date = st.date_input("Choose prediction end date", pd.to_datetime('today'))
    
    look_back = st.slider("Choose look-back period (days)", min_value=30, max_value=180, value=60, step=10)

    if st.button('Get Data and Analyze'):
        train_df = fetch_stock_data(ticker_symbol, train_start_date, train_end_date)
        if train_df.empty:
            st.error(f"Cannot retrieve {ticker_symbol} stock data, please check the input.")
        else:
            st.write(f"{ticker_symbol} stock data (Training)")
            st.write(train_df)
            plot_stock_data(train_df, ticker_symbol, "Close Price Time Series (Training)")

            # Calculate and plot returns
            train_df = calculate_returns(train_df)
            plot_returns(train_df, ticker_symbol)

            # Display basic statistics
            st.subheader("Basic Statistics")
            st.write(train_df[['Close', 'Daily_Return']].describe())

            # Linear Regression
            st.subheader("Linear Regression Prediction")
            lr_model = LinearRegressionModel()
            X_train = np.arange(len(train_df))
            lr_model.fit(X_train, train_df['Close'].values)

            future_df = fetch_stock_data(ticker_symbol, predict_start_date, predict_end_date)
            future_dates = future_df.index
            future_time = np.arange(len(train_df), len(train_df) + len(future_dates))
            lr_predicted_prices = lr_model.predict(future_time)
            lr_predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': lr_predicted_prices})

            # LSTM
            st.subheader("LSTM Prediction")
            lstm_model = LSTMModel(look_back=look_back)
            lstm_model.fit(train_df['Close'].values)
            lstm_predicted_prices = lstm_model.predict(future_df['Close'].values)
            lstm_predicted_df = pd.DataFrame({'Date': future_dates[-len(lstm_predicted_prices):], 'Predicted_Close': lstm_predicted_prices.flatten()})

            # Transformer
            st.subheader("Transformer Prediction")
            transformer_model = TransformerModel(look_back=look_back)
            transformer_model.fit(train_df['Close'].values)
            transformer_predicted_prices = transformer_model.predict(future_df['Close'].values)
            transformer_predicted_df = pd.DataFrame({'Date': future_dates[-len(transformer_predicted_prices):], 'Predicted_Close': transformer_predicted_prices.flatten()})

            if not future_df.empty:
                plot_comparison(future_df, lr_predicted_df, ticker_symbol, "Linear Regression: Actual vs Predicted")
                plot_comparison(future_df, lstm_predicted_df, ticker_symbol, "LSTM: Actual vs Predicted")
                plot_comparison(future_df, transformer_predicted_df, ticker_symbol, "Transformer: Actual vs Predicted")

                actual_prices = future_df['Close'].values

                # Linear Regression Metrics
                lr_mse, lr_rmse, lr_mae, lr_r2 = calculate_metrics(actual_prices, lr_predicted_df['Predicted_Close'].values[:len(actual_prices)])
                st.write("Linear Regression Metrics:")
                display_metrics(lr_mse, lr_rmse, lr_mae, lr_r2, ticker_symbol)

                # LSTM Metrics
                lstm_mse, lstm_rmse, lstm_mae, lstm_r2 = calculate_metrics(actual_prices[-len(lstm_predicted_prices):], lstm_predicted_prices.flatten())
                st.write("LSTM Metrics:")
                display_metrics(lstm_mse, lstm_rmse, lstm_mae, lstm_r2, ticker_symbol)

                # Transformer Metrics
                transformer_mse, transformer_rmse, transformer_mae, transformer_r2 = calculate_metrics(actual_prices[-len(transformer_predicted_prices):], transformer_predicted_prices.flatten())
                st.write("Transformer Metrics:")
                display_metrics(transformer_mse, transformer_rmse, transformer_mae, transformer_r2, ticker_symbol)

                # Volatility Analysis
                st.subheader("Volatility Analysis")
                volatility = train_df['Daily_Return'].std() * (252 ** 0.5)  # Annualized volatility
                st.write(f"Annualized Volatility: {volatility:.2%}")

                # Future Returns Prediction
                st.subheader("Future Returns Prediction")
                future_returns = (future_df['Close'].pct_change().mean() * 252)  # Annualized return
                st.write(f"Predicted Annualized Return: {future_returns:.2%}")

            else:
                st.warning(f"No actual data available for {ticker_symbol} in the prediction period for comparison.")
    else:
        st.info("Please enter the ticker symbol and click the Get Data and Analyze button.")

if __name__ == "__main__":
    main()