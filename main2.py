# main.py
from flask import Flask, request, jsonify, render_template_string
import onnxruntime as ort
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load ONNX model
model_path = "model_gru.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Parameters ---
window_size = 120

# --- Function to fetch EURUSD Close prices ---
import yfinance as yf
import pandas as pd

def fetch_eurusd_from_yfinance(period="1mo", interval="1h"):
    symbol = "EURUSD=X"
    data = yf.download(tickers=symbol, interval=interval, period=period)

    if data.empty:
        raise ValueError("No data received from Yahoo Finance.")

    data = data.reset_index()

    df = data[['Datetime', 'Close']].rename(columns={'Datetime': 'time', 'Close': 'close'})

    return df



# --- Route: Home ---
@app.route('/')
def home():
    return "Model is ready."

# --- Route: GRU Prediction Chart ---
@app.route('/gru/chart')
def gru_chart():
    # 1. Fetch live EURUSD Close data
    try:
        df = fetch_eurusd_from_yfinance(period="1mo", interval="1h")
    except Exception as e:
        return f"Error fetching data: {str(e)}"

    if df.empty:
        return "DataFrame is empty."

    close_prices = df['close'].values.reshape(-1, 1)

    # 2. Scale ALL prices first
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_prices)

    # 3. Prepare sequences from scaled data
    def create_sequences(data, window_size):
        X = []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
        return np.array(X)

    X = create_sequences(scaled_close, window_size)
    if len(X) == 0:
        return "Not enough data points for prediction."

    X = X.reshape((X.shape[0], window_size, 1))

    # 4. Make predictions
    preds = session.run([output_name], {input_name: X.astype(np.float32)})[0]
    preds_rescaled = scaler.inverse_transform(preds)

    # 5. Cut real close prices for comparison
    real_close = close_prices[window_size:]

    # 6. Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(real_close, label="Actual Close", linewidth=2)
    ax.plot(preds_rescaled, label="Predicted Close", linestyle='--')
    ax.set_title("EURUSD Close Price Prediction (GRU Model, Last 1 Month)")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()

    # 7. Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # 8. HTML Response
    html = f"""
    <html>
    <head><title>GRU Prediction Chart</title></head>
    <body>
    <h1>GRU Prediction vs Actual Close (EURUSD - Last 1 Month)</h1>
    <img src="data:image/png;base64,{image_base64}" />
    </body>
    </html>
    """

    return render_template_string(html)

# --- Run App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
