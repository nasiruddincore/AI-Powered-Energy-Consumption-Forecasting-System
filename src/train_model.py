import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from feature_engineering import create_time_series_features

def train_forecasting_model():
    data_path = "data/energy_consumption.csv"
    model_path = "models/energy_forecast_model.pkl"
    metrics_path = "outputs/metrics.json"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run simulate_data.py first.")
        
    print("🧠 Loading telemetry and generating features...")
    raw_df = pd.read_csv(data_path)
    df = create_time_series_features(raw_df)
    
    # Define Feature Sets and Target
    feature_cols = ["hour", "dayofweek", "month", "is_weekend", "lag_1h", "lag_24h", "rolling_mean_24h"]
    X = df[feature_cols]
    y = df["Energy_kWh"]
    
    # Chronological Time-Based Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"🏋️ Training Multi-Layer Perceptron (MLP) Regressor on {len(X_train)} samples...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Model Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n📊 Model Performance Summary:")
    print(f"  • MAE  (Mean Absolute Error): {mae:.2f} kWh")
    print(f"  • RMSE (Root Mean Sq Error):  {rmse:.2f} kWh")
    print(f"  • R² Score (Accuracy Metric): {r2:.4f}")
    
    # Save Model Artifacts
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    joblib.dump({"model": model, "features": feature_cols}, model_path)
    
    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2_Score": round(r2, 4)}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Trained model saved to: {model_path}")

if __name__ == "__main__":
    train_forecasting_model()