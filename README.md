# 🌍 GeoPrice Predict: Crisis-Driven Commodity Forecasting & Livelihood Impact Model

> A production-grade financial data science application that ingests live historical commodity data, simulates geopolitical and macroeconomic shocks, forecasts forward price anomalies, and translates wholesale market spikes into everyday consumer livelihood impacts.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Architecture Pipeline](#-architecture-pipeline)
3. [Project Structure](#-project-structure)
4. [Detailed Installation Steps](#-detailed-installation-steps)
   - [Option A: Local Virtual Environment](#option-a-local-virtual-environment)
   - [Option B: Docker Build & Run](#option-b-docker-build--run)
5. [How to Run](#-how-to-run)
6. [API & Data Attributions](#-api--data-attributions)

---

## 🔍 Project Overview

Global commodity markets are increasingly vulnerable to geopolitical conflicts, pandemics, recessions, and inflationary pressures. **GeoPrice Predict** bridges the gap between macroeconomic forecasting and everyday consumer reality. 

Key capabilities include:
* **Live Ingestion:** Fetches real-time historical daily data for key global commodities (Wheat, Rice, Sugar, Crude Oil, Gold, and Silver).
* **Shock Simulation Engine:** Simulates multi-variable crisis scenarios ("War", "Pandemic", "Recession", "High Inflation") across customizable severity scales (1 to 10).
* **Risk & Recovery Analytics:** Computes predicted peak price changes, shock durations, and Value at Risk (VaR 95%) tail risk metrics.
* **Livelihood Impact Mapping:** Directly maps raw wholesale price actions into consumer pain points (e.g., local transit fares, packaging, delivery logistics, and grocery inflation).

---

## 🏗️ Architecture Pipeline

The application is built using a modular, decoupled architecture:

[ Yahoo Finance API ]
│
▼
(model.py) ──► Data Ingestion & Normalization
│
├──► Macro Shock Simulation & Tail Risk (VaR 95%)
└──► Livelihood Impact Cascade Mapping
│
▼
(app.py)   ──► Streamlit Dashboard Interface
│
├──► Interactive KPI Cards
├──► Plotly Historical + 12-Month Crisis Curve
└──► Downstream Consumer Impact Breakdown


📁 Project Structure

```text
GeoPrice Predict: Crisis-Driven Commodity Forecasting & Livelihood Impact Model/
│
├── model.py           # Backend engine (yfinance data, shock simulation, livelihood mapping)
├── app.py             # Streamlit frontend dashboard and visualization UI
├── requirements.txt   # Python dependency manifest
├── Dockerfile         # Containerization configuration
└── README.md          # Comprehensive project documentation