import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Energy Forecasting System",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 30px; font-weight: bold; color: #1E88E5; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚡ AI-Powered Energy Consumption Forecasting System</p>', unsafe_allow_html=True)
st.caption("Industry-Oriented Smart Grid Demand Analytics & Optimization Dashboard")

# File Paths
MODEL_PATH = "models/energy_forecast_model.pkl"
PRED_PATH = "outputs/predictions.csv"
METRICS_PATH = "outputs/metrics.json"

# Safety Verification
if not os.path.exists(MODEL_PATH) or not os.path.exists(PRED_PATH):
    st.error("⚠️ Pipeline components not found! Please run project setup and training scripts first.")
    st.code("python src/simulate_data.py && python src/train_model.py && python src/forecast.py")
    st.stop()

# Load Data
df_preds = pd.read_csv(PRED_PATH)
df_preds["Datetime"] = pd.to_datetime(df_preds["Datetime"])

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# Sidebar - Operational Simulation Controls
st.sidebar.header("🕹️ Grid Simulation Parameters")
st.sidebar.markdown("Simulate environmental and operational adjustments on predicted load:")

temp_spike = st.sidebar.slider("Temperature Heatwave Spike (°C Offset)", 0, 10, 0)
occupancy_surge = st.sidebar.slider("Industrial Facility Surge (%)", 0, 50, 0)

# Apply Simulation Modifiers
adjusted_preds = df_preds["Predicted_Energy_kWh"] * (1 + (temp_spike * 0.02) + (occupancy_surge * 0.01))

# Metrics Overview Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model MAE", f"{metrics['MAE']} kWh")
col2.metric("Model RMSE", f"{metrics['RMSE']} kWh")
col3.metric("R² Score Accuracy", f"{metrics['R2_Score']}")
col4.metric("Peak Projected Load", f"{adjusted_preds.max():.1f} kWh")

st.divider()

# Interactive Plotly Chart
st.subheader("📈 Energy Usage: Actual vs AI Forecast (168-Hour Window)")

fig = go.Figure()

# Actual Energy Usage
fig.add_trace(go.Scatter(
    x=df_preds["Datetime"], 
    y=df_preds["Energy_kWh"],
    mode='lines',
    name='Actual Energy Load (kWh)',
    line=dict(color='#2E7D32', width=2)
))

# Predicted Energy Usage
fig.add_trace(go.Scatter(
    x=df_preds["Datetime"], 
    y=adjusted_preds,
    mode='lines',
    name='AI Forecasted Demand (kWh)',
    line=dict(color='#D81B60', width=2, dash='dash')
))

fig.update_layout(
    xaxis_title="Timeline",
    yaxis_title="Energy Consumption (kWh)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Data Table View
with st.expander("🔍 View Raw Forecast Telemetry & Residual Logs"):
    st.dataframe(df_preds, use_container_width=True)