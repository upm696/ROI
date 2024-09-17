from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)

# Load the trained models and scalers
with open('linear_regression_poly_model.pkl', 'rb') as file:
    linear_model = pickle.load(file)

with open('polynomial_features.pkl', 'rb') as file:
    poly = pickle.load(file)

with open('logistic_regression_model.pkl', 'rb') as file:
    classification_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('scaler_class.pkl', 'rb') as file:
    scaler_class = pickle.load(file)

# Load stock data for visualization purposes
df = pd.read_csv('all_stocks_5yr.csv') 
@app.route('/')
def index():
    # Get unique stock names for dropdown
    stock_names = df['Name'].unique()
    return render_template('index.html', stock_names=stock_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        open_price = float(data['open'])
        high_price = float(data['high'])
        low_price = float(data['low'])
        close_price = float(data['close'])
        volume = int(data['volume'])
        year = int(data['year'])
        month = int(data['month'])
        day = int(data['day'])
        ma_7 = float(data['ma_7'])
        ma_21 = float(data['ma_21'])
        volatility = float(data['volatility'])

        features = np.array([[open_price, high_price, low_price, close_price, volume, year, month, day, ma_7, ma_21, volatility]])

        # Scale features and transform
        features_scaled = scaler.transform(features)
        features_poly = poly.transform(features_scaled)

        # Predict ROI percentage
        predicted_roi = linear_model.predict(features_poly)
        predicted_roi_value = predicted_roi.item()

        # Predict profit or loss
        features_class_scaled = scaler_class.transform(features)
        predicted_profit_loss = classification_model.predict(features_class_scaled)[0]
        result = 'Profit' if predicted_profit_loss == 1 else 'Loss'

        return jsonify({
            'result': result,
            'roi_percentage': f"{predicted_roi_value:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    try:
        stock_name = request.json.get('stock_name')
        stock_data = df[df['Name'] == stock_name]

        if stock_data.empty:
            return jsonify({'error': 'Stock data not found'})

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Close Price', line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Bar(x=stock_data['date'], y=stock_data['volume'], name='Volume', marker_color='orange'), secondary_y=True)

        fig.update_layout(
            title=f"Stock Price and Volume Over Time for {stock_name}",
            xaxis_title="Date",
            yaxis_title="Close Price",
            yaxis2_title="Volume",
            template="plotly_white"
        )

        graphJSON = fig.to_json()
        return jsonify({'graph': graphJSON})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
