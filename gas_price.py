import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="JPMorgan Commodity Forecasting Model", layout="wide")

st.title("JPMorgan Commodity Forecasting Model")
st.write(
    "Natural gas price forecasting model built using Python, pandas, NumPy, SciPy, and Matplotlib. "
    "The model fits a trend plus seasonal regression curve to historical natural gas prices and forecasts future prices."
)

raw = """Dates,Prices
10/31/20,10.10
11/30/20,10.30
12/31/20,11.00
1/31/21,10.90
2/28/21,10.90
3/31/21,10.90
4/30/21,10.40
5/31/21,9.84
6/30/21,10.00
7/31/21,10.10
8/31/21,10.30
9/30/21,10.20
10/31/21,10.10
11/30/21,11.20
12/31/21,11.40
1/31/22,11.50
2/28/22,11.80
3/31/22,11.50
4/30/22,10.70
5/31/22,10.70
6/30/22,10.40
7/31/22,10.50
8/31/22,10.40
9/30/22,10.80
10/31/22,11.00
11/30/22,11.60
12/31/22,11.60
1/31/23,12.10
2/28/23,11.70
3/31/23,12.00
4/30/23,11.50
5/31/23,11.20
6/30/23,10.90
7/31/23,11.40
8/31/23,11.10
9/30/23,11.50
10/31/23,11.80
11/30/23,12.20
12/31/23,12.80
1/31/24,12.60
2/29/24,12.40
3/31/24,12.70
4/30/24,12.10
5/31/24,11.40
6/30/24,11.50
7/31/24,11.60
8/31/24,11.50
9/30/24,11.80"""

df = pd.read_csv(StringIO(raw))
df["Dates"] = pd.to_datetime(df["Dates"])
df = df.sort_values("Dates").reset_index(drop=True)

start = df["Dates"].iloc[0]
df["t"] = (df["Dates"].dt.year - start.year) * 12 + (df["Dates"].dt.month - start.month)

prices = df["Prices"].values
t_vals = df["t"].values

def seasonal_model(t, a, b, c, d):
    return a * t + b + c * np.sin(2 * np.pi * t / 12 + d)

params, _ = curve_fit(seasonal_model, t_vals, prices, p0=[0.05, 10, 1, 0])
a, b, c, d = params

def get_price(date_input):
    date = pd.to_datetime(date_input)
    t = (date.year - start.year) * 12 + (date.month - start.month)
    price = seasonal_model(t, a, b, c, d)
    return round(float(price), 2)

st.subheader("Historical Natural Gas Price Data")

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df["Dates"], prices, "o-", label="Actual prices")
ax1.set_title("Natural Gas Prices — Raw Data")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price ($)")
ax1.legend()
fig1.tight_layout()
st.pyplot(fig1)

st.subheader("Model Parameters")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Trend Slope", f"{a:.4f}")
col2.metric("Base Price", f"{b:.4f}")
col3.metric("Seasonal Amplitude", f"{c:.4f}")
col4.metric("Phase Shift", f"{d:.4f}")

st.subheader("Forecasted Price Estimates")

forecast_dates = {
    "January 2025": "2025-01-31",
    "June 2025": "2025-06-30",
    "September 2025": "2025-09-30",
    "December 2025": "2025-12-31",
    "March 2026": "2026-03-31",
}

forecast_data = []
for label, date in forecast_dates.items():
    forecast_data.append({"Date": label, "Estimated Price": f"${get_price(date)}"})

st.dataframe(pd.DataFrame(forecast_data), use_container_width=True)

st.subheader("Custom Forecast")

selected_date = st.date_input("Choose a future date to estimate price:", value=pd.to_datetime("2025-12-31"))
estimated_price = get_price(selected_date)
st.success(f"Estimated natural gas price for {selected_date}: ${estimated_price}")

st.subheader("Model Fit and 12-Month Forecast")

t_future = np.arange(0, max(t_vals) + 13)
future_dates = [start + pd.DateOffset(months=int(i)) for i in t_future]
predicted = seasonal_model(t_future, a, b, c, d)

fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(df["Dates"], prices, "o", label="Actual data", zorder=5)
ax2.plot(future_dates[:len(t_vals)], predicted[:len(t_vals)], linewidth=2, label="Model fit")
ax2.plot(future_dates[len(t_vals):], predicted[len(t_vals):], linestyle="--", linewidth=2, label="Extrapolation: 1 year forward")
ax2.axvline(x=df["Dates"].iloc[-1], linestyle=":", label="Data cutoff")
ax2.set_title("Natural Gas Price Model — Fit + Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price ($)")
ax2.legend()
fig2.tight_layout()
st.pyplot(fig2)

with st.expander("View Historical Dataset"):
    st.dataframe(df[["Dates", "Prices"]], use_container_width=True)

st.caption("Project based on a JPMorgan Quantitative Research virtual experience task. Built for educational and portfolio purposes.")
