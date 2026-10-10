# ⚡ AI-Powered Energy Consumption Forecasting System

An industry-oriented AI platform designed to predict hourly electrical energy usage in smart buildings, factories, and utility grids. By capturing diurnal activity cycles, temporal trends, and autoregressive lag metrics, the system delivers high-precision load forecasts to support smart city energy management and climate tech initiatives.

---

## 📌 Problem Statement
* **Grid Demand Mismatch:** Unpredictable power spikes force utility companies to activate expensive, high-emission peaker plants.
* **High Operational Costs:** Industrial facilities pay severe financial penalties when energy consumption exceeds pre-contracted peak thresholds.
* **Renewable Integration Challenges:** Solar and wind power require accurate demand predictions to coordinate battery storage (BESS) discharging.

---

## 🎯 Project Objectives
1. **Time-Series Feature Extraction:** Construct 1-hour/24-hour lag observations and temporal features from smart meter telemetry.
2. **Predictive Modeling:** Train a Multi-Layer Perceptron (MLP) Regressor to capture non-linear energy consumption patterns.
3. **Interactive Simulation UI:** Provide a web interface for real-time scenario stress-testing (e.g., heatwaves, facility load surges).

---

## 📐 System Architecture & Workflow

```text
[ Smart Meter Telemetry ] ──► [ Feature Engineering (Lags/Rolling) ]
                                            │
                                            ▼
[ Streamlit UI Dashboard ] ◄── [ MLP Neural Network Model ]