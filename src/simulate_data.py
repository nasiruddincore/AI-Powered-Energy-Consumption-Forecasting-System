import os
import numpy as np
import pandas as pd

def generate_energy_dataset(output_path="data/energy_consumption.csv", days=365):
    """
    Simulates hourly energy consumption data (in kWh) for a smart facility/grid zone.
    """
    print("⚡ Generating synthetic energy consumption dataset...")
    np.random.seed(42)
    
    # Create hourly datetime range
    date_range = pd.date_range(start="2025-01-01 00:00:00", periods=days*24, freq="h")
    df = pd.DataFrame({"Datetime": date_range})
    
    # Extract temporal components for signal generation
    hours = df["Datetime"].dt.hour
    dayofweek = df["Datetime"].dt.dayofweek
    dayofyear = df["Datetime"].dt.dayofyear
    
    # 1. Base Load
    base_load = 50.0 
    
    # 2. Daily Diurnal Pattern (Peak usage during afternoon 13:00 - 18:00)
    daily_pattern = 25.0 * np.sin(np.pi * (hours - 6) / 12)
    daily_pattern = np.maximum(0, daily_pattern)
    
    # 3. Weekly Pattern (Lower demand on weekends)
    weekend_effect = np.where(dayofweek >= 5, -15.0, 5.0)
    
    # 4. Seasonal Variation (Higher load in summer/winter due to HVAC)
    seasonal_effect = 15.0 * np.cos(2 * np.pi * (dayofyear - 170) / 365)
    
    # 5. Random Noise
    noise = np.random.normal(0, 4.0, size=len(df))
    
    # Total Energy Consumption (kWh)
    energy_kWh = base_load + daily_pattern + weekend_effect + seasonal_effect + noise
    df["Energy_kWh"] = np.round(np.maximum(10.0, energy_kWh), 2)
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset successfully created at: {output_path} ({len(df)} records)")

if __name__ == "__main__":
    generate_energy_dataset()