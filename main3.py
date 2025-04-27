# main.py
from flask import Flask, request, render_template_string
import onnxruntime as ort
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
from sklearn.preprocessing import MinMaxScaler

# --- Initialize Flask ---
app = Flask(__name__)

# --- Load ONNX model once at server start ---
model_path = "model_gru.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Parameters ---
window_size = 120
fixed_symbol = "EURUSD=X"  # Always EURUSD now

# --- Function to fetch live EURUSD data ---
def fetch_data_from_yfinance(period="1mo", interval="1h"):
    data = yf.download(tickers=fixed_symbol, interval=interval, period=period)

    if data.empty:
        raise ValueError("No data received from Yahoo Finance.")

    data = data.reset_index()

    df = data[['Datetime', 'Close']].rename(columns={'Datetime': 'time', 'Close': 'close'})
    return df

# --- Home page route ---
@app.route('/')
def home():
    return "Model is ready."

# --- GRU Prediction Chart route ---
@app.route('/gru/chart')
def gru_chart():
    # 1. Get friendly period from URL (default to 1 month)
    period_friendly = request.args.get('period', '1mo').lower()

    # Map friendly periods to yfinance periods
    period_mapping = {
        "today": ("1d", None),     # 1 day
        "3d": ("5d", 3),           # fetch 5 days, filter 3 days
        "1w": ("7d", None),        # 1 week
        "1mo": ("1mo", None),      # 1 month
        "3mo": ("3mo", None),      # 3 months
        "6mo": ("6mo", None)       # 6 months
    }

    if period_friendly not in period_mapping:
        return f"Invalid period. Choose one of: today, 3d, 1w, 1mo, 3mo, 6mo"

    yf_period, days_filter = period_mapping[period_friendly]

    interval = "1h"  # Fixed 1-hour candles

    # 2. Fetch live EURUSD data
    try:
        df = fetch_data_from_yfinance(period=yf_period, interval=interval)
    except Exception as e:
        return f"Error fetching data: {str(e)}"

    if df.empty:
        return "DataFrame is empty."

    # 3. If needed, filter to exact last N days
    if days_filter is not None:
        last_time = df['time'].max()
        df = df[df['time'] > last_time - pd.Timedelta(days=days_filter)]

    close_prices = df['close'].values.reshape(-1, 1)

    # 4. Scaling
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_prices)

    # 5. Prepare sequences
    def create_sequences(data, window_size):
        X = []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
        return np.array(X)

    X = create_sequences(scaled_close, window_size)
    if len(X) == 0:
        return "Not enough data points for prediction."

    X = X.reshape((X.shape[0], window_size, 1))

    # 6. Predict
    preds = session.run([output_name], {input_name: X.astype(np.float32)})[0]
    preds_rescaled = scaler.inverse_transform(preds)

    # 7. Time and real Close alignment
    times = df['time'].values[window_size:]
    real_close = close_prices[window_size:]

    # 8. Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, real_close, label="Actual Close", linewidth=2)
    ax.plot(times, preds_rescaled, label="Predicted Close", linestyle='--')

    ax.set_title(f"EURUSD Close Price Prediction (GRU Model, {period_friendly.upper()})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    fig.autofmt_xdate()

    # 9. Save plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # 10. Render
    html = f"""
    <html>
    <head><title>GRU Prediction Chart</title></head>
    <body>
    <h1>EURUSD GRU Prediction ({period_friendly.upper()})</h1>
    <img src="data:image/png;base64,{image_base64}" />
    </body>
    </html>
    """

    return render_template_string(html)

# --- Run app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
