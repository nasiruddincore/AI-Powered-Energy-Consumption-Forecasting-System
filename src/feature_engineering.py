import pandas as pd
import numpy as np

def create_time_series_features(df):
    """
    Transforms raw time-series data into a supervised feature matrix.
    """
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    
    # Temporal Features
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month
    df["is_weekend"] = np.where(df["dayofweek"] >= 5, 1, 0)
    
    # Lag Features (Historical Load Inputs)
    df["lag_1h"] = df["Energy_kWh"].shift(1)    # Energy usage 1 hour ago
    df["lag_24h"] = df["Energy_kWh"].shift(24)  # Energy usage same hour yesterday
    
    # Rolling Window Statistics
    df["rolling_mean_24h"] = df["Energy_kWh"].shift(1).rolling(window=24).mean()
    
    # Drop initial NaN rows created by lag operations
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

if __name__ == "__main__":
    raw_df = pd.read_csv("data/energy_consumption.csv")
    features_df = create_time_series_features(raw_df)
    print(f"✅ Feature matrix constructed. Shape: {features_df.shape}")