import plotly.graph_objects as go
import streamlit as st
from model import (
    COMMODITY_TICKERS,
    COMMODITY_UNITS,
    predict_crisis_impact,
    get_livelihood_impact,
)

# Page configuration
st.set_page_config(
    page_title="GeoPrice Predict | Crisis Forecasting & Procurement Advisor",
    page_icon="🌍",
    layout="wide",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .recommendation-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stMetric label {
        color: #8b949e !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("🌍 GeoPrice Predict")
st.subheader(
    "Crisis-Driven Commodity Forecasting & Procurement Advisor (INR)"
)
st.markdown(
    "Simulate geopolitical and macroeconomic shocks, calculate total"
    " procurement costs in Indian Rupees (₹) for your desired quantity, and get"
    " data-driven buying advice."
)
st.markdown("---")

# Sidebar Controls
st.sidebar.header("🎛️ Simulation Controls")

selected_commodity = st.sidebar.selectbox(
    "Select Commodity", options=list(COMMODITY_TICKERS.keys()), index=3
)

unit_label = COMMODITY_UNITS.get(selected_commodity, "Units")

selected_quantity = st.sidebar.number_input(
    f"Select Quantity ({unit_label})",
    min_value=1.0,
    max_value=10000.0,
    value=50.0,
    step=1.0,
)

selected_crisis = st.sidebar.selectbox(
    "Select Crisis Event Scenario",
    options=["War", "Pandemic", "Recession", "High Inflation"],
    index=0,
)

severity_scale = st.sidebar.slider(
    "Crisis Severity Scale (1 - 10)", min_value=1, max_value=10, value=5, step=1
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Adjust quantity and simulation parameters to evaluate total budget impact"
    " on Indian households/businesses."
)

# Execute Simulation Model
with st.spinner(
    f"Simulating {selected_crisis} shock on {selected_commodity}..."
):
  simulation_results = predict_crisis_impact(
      selected_commodity, selected_crisis, severity_scale
  )
  livelihood_data = get_livelihood_impact(
      selected_commodity, simulation_results["peak_pct_change"]
  )

# Calculate Total Costs based on Quantity
baseline_total_cost = simulation_results["baseline_price"] * selected_quantity
peak_total_cost = simulation_results["predicted_peak_price"] * selected_quantity
cost_difference = peak_total_cost - baseline_total_cost

# Strategic Buying Recommendation Banner
st.markdown("### 💡 AI Procurement & Buying Recommendation")
st.markdown(
    f"""
    <div class="recommendation-box">
        <h3 style="margin: 0; color: #58a6ff;">Stance: {simulation_results['recommendation']}</h3>
        <p style="margin-top: 10px; font-size: 16px; color: #c9d1d9;"><b>Rationale:</b> {simulation_results['rationale']}</p>
        <hr style="border-color: #30363d;">
        <p style="margin: 0; font-size: 15px; color: #8b949e;">
            For your selected quantity of <b>{selected_quantity:,.1f} {unit_label}</b>, waiting out this crisis could save or cost you an estimated total difference of <b style="color: {'#f85149' if cost_difference > 0 else '#3fb950'};">₹{abs(cost_difference):,.2f}</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Panel - KPI Metric Cards (Including Total Costs)
st.markdown("### 📊 Key Performance Indicators & Total Cost Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
  st.metric(
      label=f"Current Total Cost ({selected_quantity:,.0f} {unit_label})",
      value=f"₹{baseline_total_cost:,.2f}",
  )

with col2:
  st.metric(
      label=f"Projected Peak Cost",
      value=f"₹{peak_total_cost:,.2f}",
      delta=f"{simulation_results['peak_pct_change']}%",
  )

with col3:
  st.metric(
      label="Recovery Timeline",
      value=f"{simulation_results['recovery_duration_months']} Months",
  )

with col4:
  st.metric(
      label="Tail Risk (VaR 95%)",
      value=f"{simulation_results['tail_risk_var95']}%",
  )

st.markdown("---")

# Interactive Line Plot
st.markdown(
    f"### 📈 Historical Trend & 12-Month Forward {selected_crisis} Shock Curve (INR)"
)

df_plot = simulation_results["historical_and_forecast_df"]

fig = go.Figure()

historical_df = df_plot.iloc[:-12]
fig.add_trace(
    go.Scatter(
        x=historical_df.index,
        y=historical_df["Price"],
        mode="lines",
        name="Historical Trend (Per Unit)",
        line=dict(color="#58a6ff", width=2),
    )
)

forecast_df = df_plot.iloc[-13:]
fig.add_trace(
    go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Price"],
        mode="lines",
        name="12-Month Crisis Forecast (Per Unit)",
        line=dict(color="#f85149", width=2.5, dash="dash"),
    )
)

fig.update_layout(
    title=f"{selected_commodity} Unit Price Trajectory in ₹ under {selected_crisis} (Severity: {severity_scale})",
    xaxis_title="Date",
    yaxis_title="Unit Price (INR ₹)",
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
    margin=dict(l=20, r=20, t=40, b=20),
)

# Updated width parameter for Streamlit compatibility
st.plotly_chart(fig, width="stretch")

st.markdown("---")

# Downstream Impact on Daily Life
st.markdown("### 🛒 Downstream Impact on Daily Life")

col_left, col_right = st.columns([2, 1])

with col_left:
  st.markdown(f"> **Consumer Summary:** {livelihood_data['consumer_summary']}")
  st.markdown(
      "When wholesale commodity prices undergo sudden macro-shocks, the cost"
      " burden is rapidly passed down the supply chain to retail consumers in"
      " India. Below is the breakdown of cascading costs affecting household"
      " budgets:"
  )

with col_right:
  st.markdown(
      "**Key Cost Cascade Factors:**\n"
      f"- **Transit Fare Increase:** `{livelihood_data['transit_fare_increase']}`\n"
      f"- **Packaging Cost Impact:** `{livelihood_data['packaging_cost_impact']}`\n"
      f"- **Logistics & Delivery Spike:** `{livelihood_data['delivery_logistics_spike']}`"
  )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #8b949e; font-size: 14px;'>"
    "GeoPrice Predict Engine • Powered by Streamlit, Plotly & Yahoo Finance"
    "</div>",
    unsafe_allow_html=True,
)