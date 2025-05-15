# main.py
from flask import Flask, request, render_template, redirect, url_for
import onnxruntime as ort
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from datetime import datetime
import call_behavior_alpha

# --- Initialize Flask ---
app = Flask(__name__)

# --- Load ONNX models ---
model_path_price = "static/model_gru.onnx"
session_price = ort.InferenceSession(model_path_price)
input_name_price = session_price.get_inputs()[0].name
output_name_price = session_price.get_outputs()[0].name

model_path_risk = "static/model_risk.onnx"
session_risk = ort.InferenceSession(model_path_risk)
input_name_risk = session_risk.get_inputs()[0].name
output_name_risk = session_risk.get_outputs()[0].name

# --- Parameters ---
window_size = 120
TWELVE_DATA_API_KEY = 'a6cab3620d2147dbbf1bf4fede6463f1'  # replace with your actual API key
TWELVE_BASE_URL = 'https://api.twelvedata.com/time_series'

# --- Fetch EURUSD hourly prices from Twelve Data ---
def fetch_eurusd_from_twelve_data(start, end):
    params = {
        "symbol": "EUR/USD",
        "interval": "1h",
        "apikey": TWELVE_DATA_API_KEY,
        "start_date": str(start),
        "end_date": str(end),
        "format": "JSON",
        "outputsize": 5000,
        "timezone": "UTC"
    }

    response = requests.get(TWELVE_BASE_URL, params=params)
    data = response.json()

    if "values" not in data:
        raise ValueError("Error fetching data from Twelve Data. Details: " + str(data))

    df = pd.DataFrame(data["values"])
    df["time"] = pd.to_datetime(df["datetime"])
    df["close"] = df["close"].astype(float)
    df = df[["time", "close"]].sort_values("time").reset_index(drop=True)

    if df.empty:
        raise ValueError("No matching data in the given date range.")

    return df

# --- Home Page ---
@app.route('/')
def home():
    model_categories = {
        "Finance": [
            {"id": "gru-eurusd", "name": "GRU Model for EURUSD"},
            {"id": "risk-assessment", "name": "Risk Assessment Model (MLP)"}
        ],
        "Healthcare": [
            {"id": "breast-cancer-dl", "name": "Breast Cancer Detection (Deep Learning)"}
        ],
        "Behavioral":[
            {"id": "call-behavior", "name": "Call Behavior Heatmap"}
        ]
    }
    return render_template('home.html', model_categories=model_categories)

# --- Model Parameters Page ---
@app.route('/model/<model_id>', methods=['GET'])
def model_parameters(model_id):
    if model_id == "breast-cancer-dl":
        return redirect(url_for('result'))  # still static page
    elif model_id == "call-behavior":
        return redirect(url_for('run_model', model_id="call-behavior"))  # ✅ redirect to dynamic generator
    else:
        model_name = model_id.replace('-', ' ').upper()
        return render_template('parameters.html', model_id=model_id, model_name=model_name)


# --- Static Result Page ---
@app.route('/result')
def result():
    model_results = [
        {"name": "Logistic Regression", "accuracy": 95.61, "notes": "Simple and interpretable."},
        {"name": "Random Forest", "accuracy": 96.84, "notes": "Robust and versatile."},
        {"name": "Deep Learning", "accuracy": 97.89, "notes": "Best performance; higher complexity."},
        {"name": "Bayesian Network", "accuracy": 98.10, "notes": "Used in our published research."}
    ]

    return render_template("result.html",
                           model_results=model_results,
                           model_name="Breast Cancer Detection",
                           is_healthcare=True,
                           download_onnx=url_for('static', filename='model_cancer.onnx'),
                           paper_link=url_for('static', filename='breastCancer_paper.pdf'))

# --- Run Model and Show Result ---
@app.route('/model/<model_id>/run', methods=['POST', 'GET'])
def run_model(model_id):
    if model_id == "call-behavior":
        try:
            from call_behavior_alpha import generate_behavior_heatmap
            image_base64 = generate_behavior_heatmap()
            return render_template("result.html",
                                   model_name="Call Behavior Heatmap",
                                   is_behavioral_model=True,
                                   image_base64=image_base64)
        except Exception as e:
            return f"Error running behavior model: {str(e)}"

    # --- For GRU and Risk Models ---
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    if not start_date or not end_date:
        return "Start date and end date are required."

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        df = fetch_eurusd_from_twelve_data(start=start_dt, end=end_dt)
    except Exception as e:
        return f"Error fetching data: {str(e)}"

    if df.empty:
        return "DataFrame is empty after fetching."

    close_prices = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_prices)

    def create_sequences(data, window_size):
        return np.array([data[i:i + window_size] for i in range(len(data) - window_size)])

    X = create_sequences(scaled_close, window_size)
    if len(X) == 0:
        return "Not enough data points for prediction."

    X = X.reshape((X.shape[0], window_size, 1))

    if "risk" in model_id:
        preds = session_risk.run([output_name_risk], {input_name_risk: X.astype(np.float32)})[0]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(preds, color='orange', label="Predicted Risk Score")
        ax.axhline(y=0.3, color='red', linestyle='--', label="High Risk Threshold")
        ax.axhline(y=0.6, color='green', linestyle='--', label="Low Risk Threshold")
        ax.set_title(f"Risk Prediction (From {start_date} to {end_date})")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Risk Score (0-1)")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        download_link = url_for('static', filename='model_risk.onnx')
        return render_template('result.html',
                               model_name="Risk Assessment Model",
                               is_risk_model=True,
                               image_base64=image_base64,
                               download_link=download_link)

    elif "gru" in model_id:
        preds = session_price.run([output_name_price], {input_name_price: X.astype(np.float32)})[0]
        preds = scaler.inverse_transform(preds)
        actual = close_prices[window_size:window_size + len(preds)]

        rmse = round(math.sqrt(mean_squared_error(actual, preds)), 4)
        mae = round(mean_absolute_error(actual, preds), 4)
        acc = round(100 - (mae / np.mean(actual) * 100), 2)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(actual, label="Actual")
        ax.plot(preds, label="Predicted", linestyle='--')
        ax.set_title("GRU Predicted vs Actual Prices")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        download_link = url_for('static', filename='model_gru.onnx')
        return render_template('result.html',
                               model_name="GRU Model for EURUSD",
                               is_risk_model=False,
                               rmse=rmse,
                               mae=mae,
                               accuracy=acc,
                               image_base64=image_base64,
                               download_link=download_link)

    return f"Model '{model_id}' not recognized."


# --- Run the App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
