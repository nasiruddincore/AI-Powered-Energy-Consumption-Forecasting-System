import os
import joblib
import pandas as pd
from feature_engineering import create_time_series_features

def run_forecasting_pipeline():
    model_path = "models/energy_forecast_model.pkl"
    data_path = "data/energy_consumption.csv"
    output_path = "outputs/predictions.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        raise FileNotFoundError("Model or data missing. Ensure data generator and model trainer have run.")
        
    print("🔮 Running forecasting inference pipeline...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    raw_df = pd.read_csv(data_path)
    df = create_time_series_features(raw_df)
    
    # Select last 168 hours (1 week) for forecasting evaluation
    test_df = df.tail(168).copy()
    test_df["Predicted_Energy_kWh"] = model.predict(test_df[feature_cols])
    test_df["Predicted_Energy_kWh"] = test_df["Predicted_Energy_kWh"].round(2)
    test_df["Residual_Error"] = (test_df["Energy_kWh"] - test_df["Predicted_Energy_kWh"]).round(2)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_cols = ["Datetime", "Energy_kWh", "Predicted_Energy_kWh", "Residual_Error"]
    test_df[output_cols].to_csv(output_path, index=False)
    
    print(f"✅ Forecasting complete. Results exported to: {output_path}")

if __name__ == "__main__":
    run_forecasting_pipeline()