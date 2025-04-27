# main.py
from flask import Flask, request, render_template, redirect, url_for
import onnxruntime as ort
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# --- Initialize Flask ---
app = Flask(__name__)

# --- Load ONNX model once at startup ---
model_path = "../model_gru.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Parameters ---
window_size = 120
fixed_symbol = "EURUSD=X"  # Always EURUSD

# --- Function to fetch EURUSD Close prices ---
def fetch_eurusd_from_yfinance(period="1mo", interval="1h"):
    data = yf.download(tickers=fixed_symbol, interval=interval, period=period)

    if data.empty:
        raise ValueError("No data received from Yahoo Finance.")

    data = data.reset_index()

    df = data[['Datetime', 'Close']].rename(columns={'Datetime': 'time', 'Close': 'close'})
    return df

# --- Home page route with Dropdown Menu ---
@app.route('/')
def home():
    options = [1, 2, 3, 4, 5, 6]
    return render_template('home.html', options=options)

# --- GRU Prediction and Result page ---
@app.route('/gru/chart')
def gru_chart():
    # 1. Read 'months' from URL, default to 1 month
    months = request.args.get('months', default=1, type=int)

    # Validate months
    if months not in [1, 2, 3, 4, 5, 6]:
        return "Invalid months value. Please choose between 1-6."

    # Convert months to yfinance period string
    period = f"{months}mo"

    # 2. Fetch live EURUSD Close data
    try:
        df = fetch_eurusd_from_yfinance(period=period, interval="1h")
    except Exception as e:
        return f"Error fetching data: {str(e)}"

    if df.empty:
        return "DataFrame is empty."

    close_prices = df['close'].values.reshape(-1, 1)

    # 3. Scale ALL prices
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_prices)

    # 4. Prepare sequences
    def create_sequences(data, window_size):
        X = []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
        return np.array(X)

    X = create_sequences(scaled_close, window_size)
    if len(X) == 0:
        return "Not enough data points for prediction."

    X = X.reshape((X.shape[0], window_size, 1))

    # 5. Make predictions
    preds = session.run([output_name], {input_name: X.astype(np.float32)})[0]
    preds_rescaled = scaler.inverse_transform(preds)

    # 6. Real close slice for comparison
    real_close = close_prices[window_size:]

    # 7. Calculate evaluation metrics
    mse = mean_squared_error(real_close, preds_rescaled)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(real_close, preds_rescaled)

    mean_price = np.mean(real_close)
    accuracy = max(0, 100 - (mae / mean_price * 100))  # Estimated %

    # 8. Create chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(real_close, label="Actual Close", linewidth=2)
    ax.plot(preds_rescaled, label="Predicted Close", linestyle='--')
    ax.set_title(f"EURUSD Close Price Prediction (GRU Model, Last {months} Month{'s' if months > 1 else ''})")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()

    # 9. Save chart to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # 10. Render result.html
    return render_template('result.html',
                           months=months,
                           image_base64=image_base64,
                           rmse=f"{rmse:.6f}",
                           mae=f"{mae:.6f}",
                           accuracy=f"{accuracy:.2f}"
                           )

# --- Run App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
