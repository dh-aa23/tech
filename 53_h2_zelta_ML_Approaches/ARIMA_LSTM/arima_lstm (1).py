
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tqdm import tqdm
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# Load Data
data = pd.read_csv("/kaggle/input/btcdata/btcusdt_1h.csv")  # Replace with your BTCUSDT data file
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

# Helper Function: Create LSTM Dataset
def create_lstm_dataset(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Train-Test Split
look_back = 60
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

X_train, y_train = create_lstm_dataset(train_data, look_back)
X_test, y_test = create_lstm_dataset(test_data, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model with Dropout Regularization
lstm_model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Add Early Stopping
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Train LSTM Model
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stopping])

# LSTM Predictions
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions_inversed = scaler.inverse_transform(lstm_predictions)
y_test_inversed = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MSE for LSTM
lstm_mse = mean_squared_error(y_test_inversed, lstm_predictions_inversed)
print(f"LSTM Mean Squared Error: {lstm_mse}")

# Auto ARIMA Predictions
# Auto ARIMA Predictions
arima_predictions = []

for i in tqdm(range(look_back, len(test_data))):
    # Define training data for ARIMA
    train_arima = test_data[i - look_back:i].flatten()  # Use the last 'look_back' values for training

    # Train the ARIMA model
    auto_arima_model = auto_arima(
        train_arima,
        seasonal=False,
        suppress_warnings=True,
        stepwise=True,
        error_action="ignore",
        trace=False
    )

    # Predict the next value
    forecast = auto_arima_model.predict(n_periods=1)[0]

    # Scale back the forecasted value to the original range
    forecast_inversed = scaler.inverse_transform([[forecast]])[0][0]
    arima_predictions.append(forecast_inversed)


# Calculate MSE for ARIMA
arima_mse = mean_squared_error(y_test_inversed.flatten(), arima_predictions)
print(f"ARIMA Mean Squared Error (Auto ARIMA): {arima_mse}")

# Combine Predictions
test_data_actual = scaler.inverse_transform(test_data[look_back:])
data_with_predictions = pd.DataFrame({
    "Actual": test_data_actual.flatten(),
    "LSTM_Prediction": lstm_predictions_inversed.flatten(),
    "ARIMA_Prediction": arima_predictions
}, index=data.iloc[len(data) - len(test_data_actual):].index)

# Calculate 20-Period EMA
data['EMA_20'] = data['close'].ewm(span=20).mean()

# Generate Signals
signals = []
for i in range(len(data_with_predictions)):
    actual = data_with_predictions['Actual'].iloc[i]
    lstm_pred = data_with_predictions['LSTM_Prediction'].iloc[i]
    arima_pred = data_with_predictions['ARIMA_Prediction'].iloc[i]
    ema_20 = data['EMA_20'].iloc[-len(data_with_predictions) + i]

    # Entry Signals
    if lstm_pred > actual and arima_pred > actual and actual > ema_20:
        signals.append(1)  # Long Entry
    elif lstm_pred < actual and arima_pred < actual and actual < ema_20:
        signals.append(2)  # Short Entry
    # Exit Signals
    elif signals and signals[-1] == 1 and (actual >= max(lstm_pred, arima_pred) * 1.02 or actual <= max(lstm_pred, arima_pred) * 0.995):
        signals.append(-1)  # Long Exit
    elif signals and signals[-1] == 2 and (actual <= min(lstm_pred, arima_pred) * 0.98 or actual >= min(lstm_pred, arima_pred) * 1.005):
        signals.append(-2)  # Short Exit
    else:
        signals.append(0)  # No Signal

data_with_predictions['Signal'] = signals

# Save Signals to CSV
data_with_predictions.to_csv("signals.csv", index=True)

print("Signal generation complete. Signals saved to 'signals.csv'.")

