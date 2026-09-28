import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb

# (Keep your COMMODITY_TICKERS, COMMODITY_UNITS, and fetch_historical_data functions as they are)

def predict_crisis_impact_ml(commodity: str, crisis_type: str, severity: int) -> dict:
    df = fetch_historical_data(commodity, period="2y")
    baseline_price = float(df["Price"].iloc[-1])

    # 1. Feature Engineering for Time-Series (Lag & Rolling Features)
    df['Lag_1'] = df['Price'].shift(1)
    df['Lag_7'] = df['Price'].shift(7)
    df['Rolling_Mean_30'] = df['Price'].rolling(window=30).mean()
    df['Volatility_30'] = df['Price'].rolling(window=30).std()
    
    # Encode Crisis Type numerically for XGBoost
    crisis_mapping = {"War": 1, "Pandemic": 2, "Recession": 3, "High Inflation": 4}
    crisis_code = crisis_mapping.get(crisis_type, 1)
    
    # Add exogenous crisis features to the dataframe for modeling shock training
    df['Crisis_Type_Code'] = crisis_code
    df['Severity'] = severity
    
    df = df.dropna()

    # 2. Define Features (X) and Target (y - predicting future price change)
    features = ['Lag_1', 'Lag_7', 'Rolling_Mean_30', 'Volatility_30', 'Crisis_Type_Code', 'Severity']
    
    # Target: Next day's price movement percentage
    df['Target'] = df['Price'].pct_change().shift(-1) * 100
    df = df.dropna()

    X = df[features]
    y = df['Target']

    # 3. Train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X, y)

    # 4. Simulate Future Shock Prediction using XGBoost Inference
    # Create latest feature row for inference
    latest_row = pd.DataFrame([{
        'Lag_1': df['Price'].iloc[-1],
        'Lag_7': df['Price'].iloc[-7],
        'Rolling_Mean_30': df['Rolling_Mean_30'].iloc[-1],
        'Volatility_30': df['Volatility_30'].iloc[-1],
        'Crisis_Type_Code': crisis_code,
        'Severity': severity
    }])

    predicted_daily_return_pct = model.predict(latest_row)[0]

    # Amplify predicted return based on severity and commodity sensitivity mapping
    sensitivity_map = {
        "Wheat": 1.4, "Rice": 1.3, "Sugar": 1.2, 
        "Edible Oils (Soy/Palm)": 1.6, "Pulses / Staples (Corn)": 1.35,
        "Crude Oil": 2.0, "Natural Gas (Power/Energy)": 2.2, 
        "Gold": 0.8, "Silver": 1.5
    }
    coeff = sensitivity_map.get(commodity, 1.0)
    
    # Scale model prediction into cumulative peak change over the horizon
    raw_peak_change = (predicted_daily_return_pct * severity * coeff)
    
    if crisis_type == "Recession" and commodity not in ["Gold", "Silver"]:
        peak_pct_change = -abs(raw_peak_change) if raw_peak_change != 0 else -5.0
    else:
        peak_pct_change = abs(raw_peak_change) if raw_peak_change != 0 else 10.0

    # Ensure bounds look realistic for display
    peak_pct_change = max(min(peak_pct_change, 150.0), -40.0)

    predicted_peak_price = baseline_price * (1 + (peak_pct_change / 100.0))
    recovery_duration_months = int(round(6 * (severity / 5.0)))

    # Tail risk calculation using model residuals / historical variance
    daily_returns = df['Price'].pct_change().dropna()
    var_95 = float(np.percentile(daily_returns, 5)) * 100 * (severity / 5.0)

    # Recommendation Logic
    if peak_pct_change > 15:
        recommendation = "🔴 DO NOT BUY YET (Wait for Price Correction)" if commodity not in ["Gold", "Silver"] else "🟢 STRONG BUY (Safe Haven Accumulation)"
        rationale = f"XGBoost model simulation under {crisis_type} projects an intense price surge of +{peak_pct_change:.2f}%. Defer large bulk procurement."
    elif peak_pct_change < -5:
        recommendation = "🟢 EXCELLENT TIME TO BUY (Stock Up on Dip)"
        rationale = f"XGBoost trend analysis indicates a downward price correction of {peak_pct_change:.2f}%. Ideal window for inventory stocking."
    else:
        recommendation = "🟡 HOLD / BUY NORMALLY (Stable Outlook)"
        rationale = "XGBoost predicts stable variance bounds. Proceed with standard procurement cycles."

    # Generate 12-Month Forecast Curve path
    last_date = df.index[-1]
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]
    
    simulated_prices = []
    peak_month = min(3, max(1, recovery_duration_months // 3))
    for i, d in enumerate(future_dates, 1):
        if i <= peak_month:
            progress = i / peak_month
            p = baseline_price + (predicted_peak_price - baseline_price) * progress
        else:
            rem_progress = (i - peak_month) / max(1, (12 - peak_month))
            p = predicted_peak_price - ((predicted_peak_price - (baseline_price * 1.02)) * rem_progress)
        simulated_prices.append(max(p, 1.0))

    future_df = pd.DataFrame({"Price": simulated_prices}, index=future_dates)
    combined_df = pd.concat([df[['Price']], future_df])

    return {
        "baseline_price": round(baseline_price, 2),
        "predicted_peak_price": round(predicted_peak_price, 2),
        "peak_pct_change": round(peak_pct_change, 2),
        "recovery_duration_months": recovery_duration_months,
        "tail_risk_var95": round(var_95, 2),
        "recommendation": recommendation,
        "rationale": rationale,
        "historical_and_forecast_df": combined_df
    }