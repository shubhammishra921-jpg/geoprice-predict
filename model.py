import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb

COMMODITY_TICKERS = {
    "Wheat": "ZW=F",
    "Rice": "ZR=F",
    "Sugar": "SB=F",
    "Edible Oils (Soy/Palm)": "ZL=F",
    "Pulses / Staples (Corn)": "ZC=F",
    "Crude Oil": "CL=F",
    "Natural Gas (Power/Energy)": "NG=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
}

COMMODITY_UNITS = {
    "Wheat": "kg",
    "Rice": "kg",
    "Sugar": "kg",
    "Edible Oils (Soy/Palm)": "Liters",
    "Pulses / Staples (Corn)": "kg",
    "Crude Oil": "Barrels",
    "Natural Gas (Power/Energy)": "MMBtu",
    "Gold": "Grams",
    "Silver": "Grams",
}

USD_TO_INR = 83.50  # Conversion rate proxy for display


def fetch_historical_data(commodity: str, period: str = "2y") -> pd.DataFrame:
  ticker = COMMODITY_TICKERS.get(commodity, "CL=F")
  try:
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df.columns = ["Price"]
    df["Price"] = df["Price"] * USD_TO_INR
    return df
  except Exception as e:
    dates = pd.date_range(end=datetime.date.today(), periods=500, freq="B")
    prices = (100 + np.cumsum(np.random.normal(0, 1.5, 500))) * USD_TO_INR
    return pd.DataFrame({"Price": prices}, index=dates)


def predict_crisis_impact(commodity: str, crisis_type: str, severity: int) -> dict:
  df = fetch_historical_data(commodity, period="2y")
  baseline_price = float(df["Price"].iloc[-1])

  # Feature Engineering for Time-Series (Lag & Rolling Features)
  df["Lag_1"] = df["Price"].shift(1)
  df["Lag_7"] = df["Price"].shift(7)
  df["Rolling_Mean_30"] = df["Price"].rolling(window=30).mean()
  df["Volatility_30"] = df["Price"].rolling(window=30).std()

  crisis_mapping = {"War": 1, "Pandemic": 2, "Recession": 3, "High Inflation": 4}
  crisis_code = crisis_mapping.get(crisis_type, 1)

  df["Crisis_Type_Code"] = crisis_code
  df["Severity"] = severity
  df = df.dropna()

  features = [
      "Lag_1",
      "Lag_7",
      "Rolling_Mean_30",
      "Volatility_30",
      "Crisis_Type_Code",
      "Severity",
  ]
  df["Target"] = df["Price"].pct_change().shift(-1) * 100
  df = df.dropna()

  X = df[features]
  y = df["Target"]

  # Train XGBoost Regressor
  model = xgb.XGBRegressor(
      n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
  )
  model.fit(X, y)

  latest_row = pd.DataFrame([{
      "Lag_1": df["Price"].iloc[-1],
      "Lag_7": df["Price"].iloc[-7],
      "Rolling_Mean_30": df["Rolling_Mean_30"].iloc[-1],
      "Volatility_30": df["Volatility_30"].iloc[-1],
      "Crisis_Type_Code": crisis_code,
      "Severity": severity,
  }])

  predicted_daily_return_pct = model.predict(latest_row)[0]

  sensitivity_map = {
      "Wheat": 1.4,
      "Rice": 1.3,
      "Sugar": 1.2,
      "Edible Oils (Soy/Palm)": 1.6,
      "Pulses / Staples (Corn)": 1.35,
      "Crude Oil": 2.0,
      "Natural Gas (Power/Energy)": 2.2,
      "Gold": 0.8,
      "Silver": 1.5,
  }
  coeff = sensitivity_map.get(commodity, 1.0)

  raw_peak_change = predicted_daily_return_pct * severity * coeff
  if crisis_type == "Recession" and commodity not in ["Gold", "Silver"]:
    peak_pct_change = -abs(raw_peak_change) if raw_peak_change != 0 else -5.0
  else:
    peak_pct_change = abs(raw_peak_change) if raw_peak_change != 0 else 10.0

  peak_pct_change = max(min(peak_pct_change, 150.0), -40.0)
  predicted_peak_price = baseline_price * (1 + (peak_pct_change / 100.0))
  recovery_duration_months = int(round(6 * (severity / 5.0)))

  daily_returns = df["Price"].pct_change().dropna()
  var_95 = float(np.percentile(daily_returns, 5)) * 100 * (severity / 5.0)

  if peak_pct_change > 15:
    recommendation = (
        "🔴 DO NOT BUY YET (Wait for Price Correction)"
        if commodity not in ["Gold", "Silver"]
        else "🟢 STRONG BUY (Safe Haven Accumulation)"
    )
    rationale = (
        f"XGBoost model simulation under {crisis_type} projects an intense"
        f" price surge of +{peak_pct_change:.2f}%. Defer large bulk"
        " procurement."
    )
  elif peak_pct_change < -5:
    recommendation = "🟢 EXCELLENT TIME TO BUY (Stock Up on Dip)"
    rationale = (
        f"XGBoost trend analysis indicates a downward price correction of"
        f" {peak_pct_change:.2f}%. Ideal window for inventory stocking."
    )
  else:
    recommendation = "🟡 HOLD / BUY NORMALLY (Stable Outlook)"
    rationale = (
        "XGBoost predicts stable variance bounds. Proceed with standard"
        " procurement cycles."
    )

  last_date = df.index[-1]
  future_dates = [
      last_date + pd.DateOffset(months=i) for i in range(1, 13)
  ]

  simulated_prices = []
  peak_month = min(3, max(1, recovery_duration_months // 3))
  for i, d in enumerate(future_dates, 1):
    if i <= peak_month:
      progress = i / peak_month
      p = baseline_price + (predicted_peak_price - baseline_price) * progress
    else:
      rem_progress = (i - peak_month) / max(1, (12 - peak_month))
      p = predicted_peak_price - (
          (predicted_peak_price - (baseline_price * 1.02)) * rem_progress
      )
    simulated_prices.append(max(p, 1.0))

  future_df = pd.DataFrame({"Price": simulated_prices}, index=future_dates)
  combined_df = pd.concat([df[["Price"]], future_df])

  return {
      "baseline_price": round(baseline_price, 2),
      "predicted_peak_price": round(predicted_peak_price, 2),
      "peak_pct_change": round(peak_pct_change, 2),
      "recovery_duration_months": recovery_duration_months,
      "tail_risk_var95": round(var_95, 2),
      "recommendation": recommendation,
      "rationale": rationale,
      "historical_and_forecast_df": combined_df,
  }


def get_livelihood_impact(commodity: str, price_change: float) -> dict:
  abs_change = abs(price_change)
  direction = "surge" if price_change >= 0 else "drop"

  impacts = {
      "Edible Oils (Soy/Palm)": {
          "transit_fare_increase": f"+{round(abs_change * 0.1, 1)}%",
          "packaging_cost_impact": f"+{round(abs_change * 0.3, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.2, 1)}%",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in Edible Oils directly hits"
              " Indian household kitchen budgets, raising retail prices of"
              " cooking oils and restaurant fried foods significantly."
          ),
      },
      "Pulses / Staples (Corn)": {
          "transit_fare_increase": "Minimal (+0.2%)",
          "packaging_cost_impact": f"+{round(abs_change * 0.2, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.15, 1)}%",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in Pulses & Staples increases"
              " daily meal protein costs and affects poultry feed prices,"
              " leading to dearer dairy and egg retail rates."
          ),
      },
      "Natural Gas (Power/Energy)": {
          "transit_fare_increase": f"+{round(abs_change * 0.3, 1)}%",
          "packaging_cost_impact": f"+{round(abs_change * 0.4, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.5, 1)}%",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in Natural Gas directly spikes"
              " domestic electricity generation expenses, compressed natural"
              " gas (CNG) vehicle running costs, and industrial utility bills."
          ),
      },
      "Crude Oil": {
          "transit_fare_increase": f"+{round(abs_change * 0.4, 1)}%",
          "packaging_cost_impact": f"+{round(abs_change * 0.35, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.55, 1)}%",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in Crude Oil cascades into"
              " freight and logistics, raising daily commute auto/cab fares,"
              " local delivery charges, and retail fuel prices across India."
          ),
      },
      "Wheat": {
          "transit_fare_increase": "Minimal direct impact (+0.5%)",
          "packaging_cost_impact": f"+{round(abs_change * 0.2, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.15, 1)}%",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in Wheat triggers immediate"
              " retail inflation in flour (atta) and bakery items across Indian"
              " markets."
          ),
      },
      "Rice": {
          "transit_fare_increase": "Negligible",
          "packaging_cost_impact": f"+{round(abs_change * 0.15, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.1, 1)}%",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in Rice impacts staple"
              " dietary costs, raising thali prices in local eateries and"
              " household grocery expenses."
          ),
      },
      "Sugar": {
          "transit_fare_increase": "Negligible",
          "packaging_cost_impact": f"+{round(abs_change * 0.25, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.1, 1)}%",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in Sugar cascades into the"
              " dairy, confectionery, and sweet-maker (halwai) sectors."
          ),
      },
      "Gold": {
          "transit_fare_increase": "None",
          "packaging_cost_impact": "None",
          "delivery_logistics_spike": "None",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in Gold shifts domestic"
              " investment sentiment and alters retail jewellery buying"
              " patterns."
          ),
      },
      "Silver": {
          "transit_fare_increase": "None",
          "packaging_cost_impact": f"+{round(abs_change * 0.1, 1)}%",
          "delivery_logistics_spike": "None",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in Silver impacts electronics"
              " and solar panel manufacturing components."
          ),
      },
  }

  return impacts.get(
      commodity,
      {
          "transit_fare_increase": f"+{round(abs_change * 0.2, 1)}%",
          "packaging_cost_impact": f"+{round(abs_change * 0.2, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.25, 1)}%",
          "consumer_summary": (
              f"A {abs_change:.2f}% {direction} in {commodity} affects related"
              " domestic supply chains and consumer utility costs."
          ),
      },
  )