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
model_path = "model_gru.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Parameters ---
window_size = 120
fixed_symbol = "EURUSD=X"  # Always EURUSD

# --- Function to fetch EURUSD Close prices ---
def fetch_eurusd_from_yfinance(start_date, end_date, interval="1h"):
    data = yf.download(tickers=fixed_symbol, start=start_date, end=end_date, interval=interval)

    if data.empty:
        raise ValueError("No data received from Yahoo Finance.")

    data = data.reset_index()
    df = data[['Datetime', 'Close']].rename(columns={'Datetime': 'time', 'Close': 'close'})
    return df

# --- Home page ---
@app.route('/')
def home():
    models = [
        {"id": "gru-eurusd", "name": "GRU Model for EURUSD"},
        # Add more models here later
    ]
    return render_template('home.html', models=models)

# --- Model parameter form page ---
@app.route('/model/<model_id>', methods=['GET'])
def model_parameters(model_id):
    model_name = model_id.replace('-', ' ').upper()
    return render_template('parameters.html', model_id=model_id, model_name=model_name)

# --- Run model and show result ---
@app.route('/model/<model_id>/run', methods=['POST'])
def run_model(model_id):
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    if not start_date or not end_date:
        return "Start date and end date are required."

    # Fetch data for selected date range
    try:
        df = fetch_eurusd_from_yfinance(start_date=start_date, end_date=end_date)
    except Exception as e:
        return f"Error fetching data: {str(e)}"

    if df.empty:
        return "DataFrame is empty after fetching."

    close_prices = df['close'].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_prices)

    # Create sequences
    def create_sequences(data, window_size):
        X = []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
        return np.array(X)

    X = create_sequences(scaled_close, window_size)
    if len(X) == 0:
        return "Not enough data points for prediction."

    X = X.reshape((X.shape[0], window_size, 1))

    # Make predictions
    preds = session.run([output_name], {input_name: X.astype(np.float32)})[0]
    preds_rescaled = scaler.inverse_transform(preds)

    real_close = close_prices[window_size:]

    # Calculate evaluation
    mse = mean_squared_error(real_close, preds_rescaled)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(real_close, preds_rescaled)
    mean_price = np.mean(real_close)
    accuracy = max(0, 100 - (mae / mean_price * 100))  # estimated %

    # Create chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(real_close, label="Actual Close", linewidth=2)
    ax.plot(preds_rescaled, label="Predicted Close", linestyle='--')
    ax.set_title(f"Prediction (From {start_date} to {end_date})")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()

    # Save chart
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    model_name = model_id.replace('-', ' ').upper()

    return render_template('result.html',
        model_name=model_name,
        image_base64=image_base64,
        rmse=f"{rmse:.6f}",
        mae=f"{mae:.6f}",
        accuracy=f"{accuracy:.2f}"
    )

# --- Run the App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
