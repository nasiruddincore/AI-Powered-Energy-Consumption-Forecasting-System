# ⚡ AI-Powered Energy Consumption Forecasting System

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Frontend](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

An industry-oriented AI platform designed to predict hourly electrical energy usage in smart buildings, factories, and utility grids. By capturing diurnal activity cycles, temporal trends, and autoregressive lag metrics, the system delivers high-precision load forecasts to support smart city energy management and climate tech initiatives.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [System Objectives](#-system-objectives)
- [Industry Relevance & Use Cases](#-industry-relevance--use-cases)
- [System Architecture & Data Flow](#-system-architecture--data-flow)
- [Folder Structure](#-folder-structure)
- [Installation & Setup Guide](#-installation--setup-guide)
- [How to Run (Execution Workflow)](#-how-to-run-execution-workflow)
- [Model Evaluation Results](#-model-evaluation-results)
- [Author & License](#-author--license)

---

## 📌 Project Overview
Energy demand fluctuates every hour depending on human activity, weather conditions, and industrial operations. This project uses a Multi-Layer Perceptron (MLP) neural network to study historical energy data and forecast future power requirements. The trained model is deployed through an interactive web dashboard, enabling facility managers and grid operators to run dynamic scenario simulations.

---

## ⚠️ Problem Statement
* **Grid Demand Mismatch:** Unpredictable power spikes force utility companies to activate expensive, high-emission peaker plants.
* **High Operational Costs:** Industrial facilities pay severe financial penalties when energy consumption exceeds pre-contracted peak thresholds.
* **Hardware Barriers:** Accessing live physical smart grid data is restricted for students and researchers. This project solves this via a realistic time-series telemetry simulation engine.

---

## 🎯 System Objectives
1. **Virtual Telemetry Generation:** Simulate a realistic, 1-year continuous hourly smart meter dataset incorporating baseloads, diurnal patterns, and seasonal noise.
2. **Time-Series Feature Extraction:** Construct 1-hour/24-hour lag observations and rolling averages to capture temporal human behavior.
3. **Predictive Modeling:** Train a Scikit-learn MLP Regressor to predict future energy loads with an $R^2$ accuracy score above 0.90.
4. **Interactive Simulation UI:** Provide a Streamlit dashboard for real-time scenario stress-testing (e.g., simulating heatwaves or factory surges).

---

## 🏭 Industry Relevance & Use Cases
* **Smart Cities & Utilities:** Balances power supply with fluctuating demand across grid regions to prevent blackouts.
* **Manufacturing Plants:** Schedules energy-heavy production shifts during off-peak tariff hours to reduce peak-demand charges.
* **Data Centers:** Dynamically scales HVAC cooling systems up or down based on predicted computational server load and thermal output.

---

## 📐 System Architecture & Data Flow

```text
[ Virtual Data Generator ] ──(Simulated Smart Grid Logs)──► [ data/energy_consumption.csv ]
                                                                      │
                                                           (Feature Engineering)
                                                                      ▼
[ Streamlit UI Dashboard ] ◄──────(Inference)────────────── [ MLP Neural Network Model ]
