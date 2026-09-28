import datetime
import numpy as np
import pandas as pd
import yfinance as yf

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

  crisis_multipliers = {
      "War": {"peak_factor": 0.08, "duration_base": 6, "tail_risk": 1.8},
      "Pandemic": {"peak_factor": 0.06, "duration_base": 8, "tail_risk": 1.5},
      "Recession": {"peak_factor": -0.05, "duration_base": 10, "tail_risk": 2.2},
      "High Inflation": {
          "peak_factor": 0.07,
          "duration_base": 12,
          "tail_risk": 1.6,
      },
  }

  coeff = sensitivity_map.get(commodity, 1.0)
  c_params = crisis_multipliers.get(
      crisis_type, {"peak_factor": 0.05, "duration_base": 6, "tail_risk": 1.5}
  )

  raw_change = severity * c_params["peak_factor"] * coeff
  if crisis_type == "Recession" and commodity not in ["Gold", "Silver"]:
    peak_pct_change = -abs(raw_change) * 100
  else:
    peak_pct_change = abs(raw_change) * 100

  predicted_peak_price = baseline_price * (1 + (peak_pct_change / 100.0))
  recovery_duration_months = int(
      round(c_params["duration_base"] * (severity / 5.0))
  )

  daily_returns = df["Price"].pct_change().dropna()
  var_95 = (
      float(np.percentile(daily_returns, 5))
      * 100
      * c_params["tail_risk"]
      * (severity / 5.0)
  )

  if peak_pct_change > 15:
    recommendation = (
        "🔴 DO NOT BUY YET (Wait for Price Correction)"
        if commodity not in ["Gold", "Silver"]
        else "🟢 STRONG BUY (Safe Haven Accumulation)"
    )
    rationale = (
        f"The simulated {crisis_type} drives a sharp price surge of"
        f" +{peak_pct_change:.2f}%. Purchasing large quantities now will"
        " over-leverage your budget. It is advised to buy only immediate"
        " essentials and wait."
    )
  elif peak_pct_change < -5:
    recommendation = "🟢 EXCELLENT TIME TO BUY (Stock Up on Dip)"
    rationale = (
        f"Prices are projected to fall by {peak_pct_change:.2f}% due to"
        f" {crisis_type}. This creates a profitable procurement window to"
        " stockpile inventory at lower rates."
    )
  else:
    recommendation = "🟡 HOLD / BUY NORMALLY (Stable Outlook)"
    rationale = (
        "Price volatility is within standard bounds. Proceed with routine"
        " purchases according to normal consumption needs without panic buying."
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
  combined_df = pd.concat([df, future_df])

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
              f"A {abs_change}% {direction} in Edible Oils directly hits Indian"
              " household kitchen budgets, raising retail prices of cooking oils"
              " and restaurant fried foods significantly."
          ),
      },
      "Pulses / Staples (Corn)": {
          "transit_fare_increase": "Minimal (+0.2%)",
          "packaging_cost_impact": f"+{round(abs_change * 0.2, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.15, 1)}%",
          "consumer_summary": (
              f"A {abs_change}% {direction} in Pulses & Staples increases daily"
              " meal protein costs and affects poultry feed prices, leading to"
              " dearer dairy and egg retail rates."
          ),
      },
      "Natural Gas (Power/Energy)": {
          "transit_fare_increase": f"+{round(abs_change * 0.3, 1)}%",
          "packaging_cost_impact": f"+{round(abs_change * 0.4, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.5, 1)}%",
          "consumer_summary": (
              f"A {abs_change}% {direction} in Natural Gas directly spikes"
              " domestic electricity generation expenses, compressed natural"
              " gas (CNG) vehicle running costs, and industrial utility bills."
          ),
      },
      "Crude Oil": {
          "transit_fare_increase": f"+{round(abs_change * 0.4, 1)}%",
          "packaging_cost_impact": f"+{round(abs_change * 0.35, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.55, 1)}%",
          "consumer_summary": (
              f"A {abs_change}% {direction} in Crude Oil cascades into freight"
              " and logistics, raising daily commute auto/cab fares, local"
              " delivery charges, and retail fuel prices across India."
          ),
      },
      "Wheat": {
          "transit_fare_increase": "Minimal direct impact (+0.5%)",
          "packaging_cost_impact": f"+{round(abs_change * 0.2, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.15, 1)}%",
          "consumer_summary": (
              f"A {abs_change}% {direction} in Wheat triggers immediate retail"
              " inflation in flour (atta) and bakery items across Indian"
              " markets."
          ),
      },
      "Rice": {
          "transit_fare_increase": "Negligible",
          "packaging_cost_impact": f"+{round(abs_change * 0.15, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.1, 1)}%",
          "consumer_summary": (
              f"A {abs_change}% {direction} in Rice impacts staple dietary"
              " costs, raising thali prices in local eateries and household"
              " grocery expenses."
          ),
      },
      "Sugar": {
          "transit_fare_increase": "Negligible",
          "packaging_cost_impact": f"+{round(abs_change * 0.25, 1)}%",
          "delivery_logistics_spike": f"+{round(abs_change * 0.1, 1)}%",
          "consumer_summary": (
              f"A {abs_change}% {direction} in Sugar cascades into the dairy,"
              " confectionery, and sweet-maker (halwai) sectors."
          ),
      },
      "Gold": {
          "transit_fare_increase": "None",
          "packaging_cost_impact": "None",
          "delivery_logistics_spike": "None",
          "consumer_summary": (
              f"A {abs_change}% {direction} in Gold shifts domestic investment"
              " sentiment and alters retail jewellery buying patterns."
          ),
      },
      "Silver": {
          "transit_fare_increase": "None",
          "packaging_cost_impact": f"+{round(abs_change * 0.1, 1)}%",
          "delivery_logistics_spike": "None",
          "consumer_summary": (
              f"A {abs_change}% {direction} in Silver impacts electronics and"
              " solar panel manufacturing components."
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
              f"A {abs_change}% {direction} in {commodity} affects related"
              " domestic supply chains and consumer utility costs."
          ),
      },
  )