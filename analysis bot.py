import requests
import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import logging
import joblib
import os
import ta
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
SEQUENCE_LENGTH = 50
EPOCHS = 20
BATCH_SIZE = 32
MIN_REQUIRED_ROWS = 30  # Minimum rows required for SMA, RSI, etc.


# -----------------------------------
# 1. Fetch Data from APIs
# -----------------------------------
def simulate_gbm(initial_price, days, drift, volatility, simulations=1000):
       """Simulate price paths using Geometric Brownian Motion."""
       dt = 1 / 365  # Daily time step
       paths = np.zeros((simulations, days))
       paths[:, 0] = initial_price

       for t in range(1, days):
           z = np.random.standard_normal(simulations)  # Random normal values
           paths[:, t] = paths[:, t - 1] * np.exp(
               (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
           )
       return paths
def calculate_gbm_risk(paths):
    """Calculate risk metrics from GBM simulated paths."""
    final_prices = paths[:, -1]
    VaR_95 = np.percentile(final_prices, 5)  # 5th percentile
    expected_shortfall = final_prices[final_prices < VaR_95].mean()
    return VaR_95, expected_shortfall

### Updated Code Implementation
def calculate_risk_metrics(data):
    """Calculate Sharpe Ratio, VaR, and Expected Shortfall."""
    logging.info(f"Input to calculate_risk_metrics: Shape={data.shape}, Columns={list(data.columns)}")
    
    if data.empty or "price" not in data.columns or data["price"].isnull().any():
        logging.warning("Invalid data for risk metrics. Skipping calculation.")
        return None, None, None, None, None  # Return None for all metrics if data is invalid

    daily_returns = data["price"].pct_change().dropna()

    if daily_returns.empty:
        logging.warning("Insufficient returns data for calculating risk metrics. Skipping calculation.")
        return None, None, None, None, None

    try:
        mean_return = daily_returns.mean()
        std_dev_return = daily_returns.std()
        sharpe_ratio = mean_return / std_dev_return if std_dev_return > 0 else 0

        gbm_paths = simulate_gbm(data["price"].iloc[-1], 30, mean_return, std_dev_return)
        VaR_95, expected_shortfall = calculate_gbm_risk(gbm_paths)

        return sharpe_ratio, VaR_95, expected_shortfall, mean_return, std_dev_return
    except Exception as e:
        logging.error(f"Error calculating risk metrics: {e}")
        return None, None, None, None, None  # Return None for all metrics if an error occurs

def calculate_gbm_risk(paths):
    """Calculate risk metrics from GBM simulated paths."""
    final_prices = paths[:, -1]
    VaR_95 = np.percentile(final_prices, 5)  # 5th percentile
    expected_shortfall = final_prices[final_prices < VaR_95].mean()
    return VaR_95, expected_shortfall

def fetch_top_coins(limit=100):
    """Fetch the top coins from CoinGecko."""
    cg = CoinGeckoAPI()
    try:
        coins = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=limit, page=1)
        coin_ids = [coin['id'] for coin in coins]
        logging.info(f"Fetched top {len(coin_ids)} coins from CoinGecko.")
        return coin_ids
    except Exception as e:
        logging.error(f"Failed to fetch coins from CoinGecko: {e}")
        return []

def merge_data(price_data, sentiment_data):
    """Merge price and sentiment data on date."""
    merged = pd.merge(price_data, sentiment_data, on='date', how='left')
    merged['average_sentiment'] = merged['average_sentiment'].ffill()  # Forward fill sentiment values
    logging.info(f"Merged data shape: {merged.shape}, Columns: {list(merged.columns)}")
    return merged

def fetch_senticrypt_data():
    """Fetch historical market sentiment from SentiCrypt."""
    url = "https://api.senticrypt.com/v2/all.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        sentiment_data = response.json()

        # Log the raw sentiment data
        logging.info(f"Raw sentiment data fetched: {sentiment_data}")

        records = []
        for record in sentiment_data:
            records.append({
                "date": datetime.strptime(record['date'], "%Y-%m-%d").date(),
                "average_sentiment": record.get("mean", 0),  # Use the mean sentiment score
                "sentiment_sum": record.get("sum", 0),
                "sentiment_count": record.get("count", 0),
                "score1": record.get("score1", 0),  # Optional: include other scores if needed
                "score2": record.get("score2", 0),
                "score3": record.get("score3", 0),
            })
        df = pd.DataFrame(records)

        logging.info(f"Fetched sentiment data: Shape={df.shape}, Columns={list(df.columns)}")
        logging.info(f"Null values in sentiment data:\n{df.isnull().sum()}")
        logging.info(f"Sample sentiment data:\n{df.head()}")

        # Check if all average_sentiment values are zero
        if (df['average_sentiment'] == 0).all():
            logging.warning("All average_sentiment values are zero.")
        else:
            logging.info("Some average_sentiment values are non-zero.")

        return df
    except Exception as e:
        logging.error(f"Failed to fetch SentiCrypt data: {e}")
        return pd.DataFrame()

def fetch_historical_data(coin_id, days=365):
    """Fetch historical price and volume data from CoinGecko."""
    cg = CoinGeckoAPI()
    try:
        market_data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency="usd", days=days)
        prices = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
        volumes = pd.DataFrame(market_data["total_volumes"], columns=["timestamp", "volume"])
        merged = pd.merge(prices, volumes, on="timestamp")
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], unit="ms")
        merged["date"] = merged["timestamp"].dt.date
        logging.info(f"Fetched historical data for {coin_id}: Shape={merged.shape}, Columns={list(merged.columns)}")
        logging.info(f"Null values in historical data for {coin_id}:\n{merged.isnull().sum()}")
        logging.info(f"Sample historical data for {coin_id}:\n{merged.head()}")
        return merged
    except Exception as e:
        logging.error(f"Failed to fetch historical data for {coin_id}: {e}")
        return pd.DataFrame()

def fetch_live_data_with_history(coin_list, sentiment_data):
    """Fetch live data and combine it with recent historical data for RNN prediction."""
    cg = CoinGeckoAPI()
    live_data_with_history = []

    for coin in coin_list:
        historical_data = fetch_historical_data(coin_id=coin, days=100)
        if historical_data.empty or "price" not in historical_data.columns:
            logging.warning(f"No sufficient historical data for {coin}. Skipping.")
            continue

        logging.info(f"Historical data for {coin}: Shape={historical_data.shape}, Columns={list(historical_data.columns)}")

        try:
            market_data = cg.get_price(
                ids=coin,
                vs_currencies="usd",
                include_market_cap=True,
                include_24hr_vol=True
            )[coin]
            live_row = {
                "date": pd.Timestamp.now().normalize(),
                "price": market_data.get("usd", 0),
                "volume": market_data.get("usd_24h_vol", 0),
                "average_sentiment": sentiment_data["average_sentiment"].mean() if not sentiment_data.empty else 0,
            }
        except Exception as e:
            logging.error(f"Error fetching live data for {coin}: {e}")
            continue

        logging.info(f"Live data for {coin}: {live_row}")

        live_row_df = pd.DataFrame([live_row])
        combined_data = pd.concat([historical_data, live_row_df], ignore_index=True)

        if combined_data["price"].isnull().any() or len(combined_data) < 50:
            logging.warning(f"Combined data for {coin} is insufficient for analysis. Skipping.")
            continue

        combined_data = calculate_indicators(combined_data)

        # Calculate risk metrics, but handle potential errors
        try:
            sharpe, var_95, es, mean_return, std_dev_return = calculate_risk_metrics(combined_data[-50:])
            combined_data["sharpe_ratio"] = sharpe
            combined_data["VaR_95"] = var_95
            combined_data["expected_shortfall"] = es
            combined_data["mean_return"] = mean_return
            combined_data["std_dev_return"] = std_dev_return
        except Exception as e:
            logging.error(f"Error calculating risk metrics for {coin}: {e}")
            combined_data["sharpe_ratio"] = None
            combined_data["VaR_95"] = None
            combined_data["expected_shortfall"] = None
            combined_data["mean_return"] = None
            combined_data["std_dev_return"] = None

        live_data_with_history.append(combined_data)

    return live_data_with_history

# -----------------------------------
# 1.2. Save/Load Models
# -----------------------------------

def save_model(model, filename):
    """Save a trained model to disk."""
    model.save(filename)
    logging.info(f"Model saved: {filename}")

def load_model_safe(filename):
    """Load a saved model, if available."""
    if os.path.exists(filename):
        logging.info(f"Loading model from: {filename}")
        return load_model(filename)
    logging.info(f"Model file not found: {filename}. Training a new model.")
    return None

# -----------------------------------
# 1.5. Risk Metrics Calculations
# -----------------------------------

def calculate_risk_metrics(data):
       """Calculate Sharpe Ratio, VaR, and Expected Shortfall."""
       logging.info(f"Input to calculate_risk_metrics: Shape={data.shape}, Columns={list(data.columns)}")
       
       if data.empty or "price" not in data.columns or data["price"].isnull().any():
           logging.warning("Invalid data for risk metrics. Skipping calculation.")
           return None, None, None, None, None  # Return None for all metrics if data is invalid

       daily_returns = data["price"].pct_change().dropna()

       if daily_returns.empty:
           logging.warning("Insufficient returns data for calculating risk metrics. Skipping calculation.")
           return None, None, None, None, None

       try:
           mean_return = daily_returns.mean()
           std_dev_return = daily_returns.std()
           sharpe_ratio = mean_return / std_dev_return if std_dev_return > 0 else 0

           gbm_paths = simulate_gbm(data["price"].iloc[-1], 30, mean_return, std_dev_return)
           VaR_95, expected_shortfall = calculate_gbm_risk(gbm_paths)

           return sharpe_ratio, VaR_95, expected_shortfall, mean_return, std_dev_return
       except Exception as e:
           logging.error(f"Error calculating risk metrics: {e}")
           return None, None, None, None, None  # Return None for all metrics if an error occurs

def calculate_indicators(data):
    """Calculate technical indicators and add them to the DataFrame."""
    if len(data) < MIN_REQUIRED_ROWS:
        logging.warning(f"Insufficient data for technical indicators. Rows available: {len(data)}. Skipping indicator calculation.")
        return data  # Return unprocessed data
    
    try:
        # SMA (Simple Moving Averages)
        data['SMA_10'] = data['price'].rolling(window=10).mean()
        data['SMA_30'] = data['price'].rolling(window=30).mean()

        # EMA (Exponential Moving Average)
        data['EMA_10'] = data['price'].ewm(span=10, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = data['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        rolling_mean = data['price'].rolling(window=20).mean()
        rolling_std = data['price'].rolling(window=20).std()
        data['bollinger_hband'] = rolling_mean + (rolling_std * 2)
        data['bollinger_lband'] = rolling_mean - (rolling_std * 2)

        # ATR (Average True Range)
        high_low = data['price'].rolling(window=14).max() - data['price'].rolling(window=14).min()
        data['ATR'] = high_low / data['price']

        # Lagged Features
        data['price_lag_1'] = data['price'].shift(1)
        data['volume_lag_1'] = data['volume'].shift(1)

        logging.info(f"Indicators calculated: Shape={data.shape}, Columns={list(data.columns)}")
        return data

    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return data  # Return unprocessed data

# -----------------------------------
# 2. Train RNN for Profitability
# -----------------------------------

def train_rnn(data, target_column, input_columns):
    """Train an RNN model with the given data."""
    features = data[input_columns]
    target = data[target_column]

    if features.isnull().any().any():
        raise ValueError("Input features contain missing values.")
    if target.isnull().any():
        raise ValueError("Target column contains missing values.")

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, "profit_scaler.pkl")

    X = np.array([features_scaled[i:i + SEQUENCE_LENGTH] for i in range(len(features_scaled) - SEQUENCE_LENGTH)])
    y = target[SEQUENCE_LENGTH:].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dense(1, activation='linear', kernel_regularizer=l2(0.01))
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Calculate RMSE for training and testing data
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    logging.info(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

    return model, scaler

# -----------------------------------
# 3. Train RNN for Risk
# -----------------------------------

def train_rnn_risk(data, target_column, input_columns):
    """Train an RNN model for risk prediction."""
    features = data[input_columns].values
    target = data[target_column].values

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, "risk_scaler.pkl")

    X, y = [], []
    for i in range(len(features_scaled) - SEQUENCE_LENGTH):
        X.append(features_scaled[i : i + SEQUENCE_LENGTH])
        y.append(target[i + SEQUENCE_LENGTH])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X.shape[2])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)

    # Calculate RMSE for risk predictions
    val_predictions = model.predict(X)
    val_rmse = np.sqrt(mean_squared_error(y, val_predictions))

    logging.info(f"Validation RMSE for risk model: {val_rmse:.4f}")

    return model, scaler

# -----------------------------------
# 4. Live Data Processing
# -----------------------------------

def predict_with_rnn(model, scaler, live_data_with_history, input_columns):
    """Predict profitability or risk using combined historical and live data."""
    predictions = []
    completeness_scores = []

    for data in live_data_with_history:
        missing_columns = [col for col in input_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing columns for prediction: {missing_columns}. Skipping this coin.")
            predictions.append(None)
            completeness_scores.append(0)
            continue

        data = data[input_columns]
        logging.info(f"Data passed to RNN for prediction: Shape={data.shape}, Columns={list(data.columns)}")

        total_values = data.size
        missing_values = data.isnull().sum().sum()
        completeness = ((total_values - missing_values) / total_values) * 100
        completeness_scores.append(completeness)

        data = data.fillna(data.mean())

        try:
            features = scaler.transform(data)
        except Exception as e:
            logging.error(f"Error scaling data: {e}")
            predictions.append(None)
            continue

        if len(features) < SEQUENCE_LENGTH:
            padded_features = np.zeros((SEQUENCE_LENGTH, features.shape[1]))
            padded_features[-len(features):] = features
            sequence = padded_features
        else:
            sequence = features[-SEQUENCE_LENGTH:]

        sequence = np.expand_dims(sequence, axis=0)

        try:
            prediction = model.predict(sequence)
            predictions.append(prediction.flatten()[0])
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            predictions.append(None)

    return predictions, completeness_scores

# -----------------------------------
# 5. Telegram Integration
# -----------------------------------

def send_telegram_message(token, chat_id, message):
    """Create BonkBot Telegram Bot swap your token and chat id"""
    """Send a message to Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

# -----------------------------------
# Main Execution
# -----------------------------------

if __name__ == "__main__":
    logging.info("Starting the execution pipeline...")

    sentiment_data = fetch_senticrypt_data()
    if sentiment_data.empty:
        logging.error("Failed to fetch sentiment data. Exiting.")
        exit()

    # Fetch all coins dynamically
    coin_list = fetch_top_coins(limit=100)
    if not coin_list:
        logging.error("No coins available for analysis. Exiting.")
        exit()

    """These metrics dont have data when I pull from the API, new metrics required"""
    input_columns = [
        "price", "volume", "average_sentiment", "SMA_10", "SMA_30", "EMA_10",
        "RSI", "bollinger_hband", "bollinger_lband", "ATR", "price_lag_1", "volume_lag_1",
        "sharpe_ratio", "VaR_95", "expected_shortfall", "mean_return", "std_dev_return"
    ]

    rnn_profit = load_model_safe("rnn_profit_model.h5")
    rnn_risk = load_model_safe("rnn_risk_model.h5")
    profit_scaler = joblib.load("profit_scaler.pkl") if os.path.exists("profit_scaler.pkl") else None
    risk_scaler = joblib.load("risk_scaler.pkl") if os.path.exists("risk_scaler.pkl") else None

    if rnn_profit is None or rnn_risk is None:
        merged_dataframes = []

        for coin in coin_list:
            price_data = fetch_historical_data(coin_id=coin, days=365)
            if price_data.empty:
                logging.warning(f"No historical price data for {coin}. Skipping.")
                continue

            try:
                merged_data = merge_data(price_data, sentiment_data)
                logging.info(f"Merged data shape: {merged_data.shape}")
                logging.info(f"Merged data preview:\n{merged_data.head()}")

                merged_data = calculate_indicators(merged_data)

                sharpe, var_95, es, mean_return, std_dev_return = calculate_risk_metrics(merged_data)
                merged_data["sharpe_ratio"] = sharpe
                merged_data["VaR_95"] = var_95
                merged_data["expected_shortfall"] = es
                merged_data["mean_return"] = mean_return
                merged_data["std_dev_return"] = std_dev_return

                merged_data["target_risk"] = 0.4 * sharpe - 0.4 * var_95 - 0.2 * es
                merged_data["target_profit"] = merged_data["price"].shift(-1) - merged_data["price"]

                merged_data.dropna(inplace=True)
                if merged_data.empty:
                    logging.warning(f"Processed data for {coin} is empty. Skipping.")
                    continue

                merged_dataframes.append(merged_data)
            except Exception as e:
                logging.error(f"Error processing data for {coin}: {e}")

        if not merged_dataframes:
            logging.error("No data available for training. Exiting.")
            exit()

        all_data = pd.concat(merged_dataframes)

        rnn_profit, profit_scaler = train_rnn(all_data, "target_profit", input_columns)
        save_model(rnn_profit, "rnn_profit_model.h5")
        joblib.dump(profit_scaler, "profit_scaler.pkl")
        logging.info("Profitability model trained and saved.")

        rnn_risk, risk_scaler = train_rnn_risk(all_data, "target_risk", input_columns)
        save_model(rnn_risk, "rnn_risk_model.h5")
        joblib.dump(risk_scaler, "risk_scaler.pkl")
        logging.info("Risk model trained and saved.")

    live_data_with_history = fetch_live_data_with_history(coin_list, sentiment_data)
    if not live_data_with_history:
        logging.error("No live data available for predictions. Exiting.")
        exit()

    profit_predictions, profit_completeness = predict_with_rnn(rnn_profit, profit_scaler, live_data_with_history, input_columns)
    risk_predictions, risk_completeness = predict_with_rnn(rnn_risk, risk_scaler, live_data_with_history, input_columns)

    profitable_coins = []
    for i, data in enumerate(live_data_with_history):
        if profit_predictions[i] is None or risk_predictions[i] is None:
            continue

        row = data.iloc[-1]
        row["predicted_profit"] = profit_predictions[i]
        row["predicted_risk"] = risk_predictions[i]
        row["profit_data_completeness"] = profit_completeness[i]
        row["risk_data_completeness"] = risk_completeness[i]
        
        if row["profit_data_completeness"] < 80 or row["risk_data_completeness"] < 80:
            continue

        confidence_level = min(10, max(1, int((1 - row["predicted_risk"]) * 10)))

        if row["predicted_profit"] > (row["predicted_risk"] * 1.5):
            profitable_coins.append({
                "coin": row["coin"],
                "predicted_profit": row["predicted_profit"],
                "predicted_risk": row["predicted_risk"],
                "confidence": confidence_level,
                "profit_data_completeness": row["profit_data_completeness"],
                "risk_data_completeness": row["risk_data_completeness"]
            })

            message = (
                f"🚀 **Trade Signal**\n"
                f"**Coin:** {row['coin']}\n"
                f"**Predicted Profit:** {row['predicted_profit']:.4f}\n"
                f"**Predicted Risk:** {row['predicted_risk']:.4f}\n"
                f"**Confidence Level:** {confidence_level}/10\n"
                f"**Profit Data Completeness:** {row['profit_data_completeness']:.2f}%\n"
                f"**Risk Data Completeness:** {row['risk_data_completeness']:.2f}%"
            )
            send_telegram_message("YOUR_TELEGRAM_BOT_TOKEN", "YOUR_CHAT_ID", message)

    if profitable_coins:
        profitable_coins_df = pd.DataFrame(profitable_coins)
        profitable_coins_df.to_csv("profitable_coins.csv", index=False)
        logging.info(f"Profitable coins saved: {len(profitable_coins)} entries.")
    else:
        logging.info("No profitable trades identified.")