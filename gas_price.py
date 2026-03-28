import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import StringIO

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
df['Dates'] = pd.to_datetime(df['Dates'], format='mixed')
df = df.sort_values('Dates').reset_index(drop=True)

start = df['Dates'].iloc[0]
df['t'] = (df['Dates'].dt.year - start.year) * 12 + \
          (df['Dates'].dt.month - start.month)

prices = df['Prices'].values
t_vals = df['t'].values

plt.figure(figsize=(12, 5))
plt.plot(df['Dates'], prices, 'o-', color='steelblue', label='Actual prices')
plt.title('Natural Gas Prices — Raw Data')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.show()

def seasonal_model(t, a, b, c, d):
    return a * t + b + c * np.sin(2 * np.pi * t / 12 + d)

params, _ = curve_fit(seasonal_model, t_vals, prices, p0=[0.05, 10, 1, 0])
a, b, c, d = params

print("=" * 40)
print("Model Parameters:")
print(f"  Trend slope (a) : {a:.4f}")
print(f"  Base price  (b) : {b:.4f}")
print(f"  Seasonal amp(c) : {c:.4f}")
print(f"  Phase shift (d) : {d:.4f}")
print("=" * 40)

def get_price(date_str):
    date = pd.to_datetime(date_str)
    t = (date.year - start.year) * 12 + (date.month - start.month)
    price = seasonal_model(t, a, b, c, d)
    return round(price, 2)

print("\nPrice Estimates:")
print(f"  January   2025 : ${get_price('2025-01-31')}")
print(f"  June      2025 : ${get_price('2025-06-30')}")
print(f"  September 2025 : ${get_price('2025-09-30')}")
print(f"  December  2025 : ${get_price('2025-12-31')}")
print(f"  March     2026 : ${get_price('2026-03-31')}")

t_future = np.arange(0, max(t_vals) + 13)
future_dates = [start + pd.DateOffset(months=int(i)) for i in t_future]
predicted = seasonal_model(t_future, a, b, c, d)

plt.figure(figsize=(14, 6))
plt.plot(df['Dates'], prices, 'o', color='steelblue', label='Actual data', zorder=5)
plt.plot(future_dates[:len(t_vals)], predicted[:len(t_vals)],
         color='orange', linewidth=2, label='Model fit')
plt.plot(future_dates[len(t_vals):], predicted[len(t_vals):],
         color='red', linestyle='--', linewidth=2, label='Extrapolation (1 year forward)')
plt.axvline(x=df['Dates'].iloc[-1], color='gray', linestyle=':', label='Data cutoff')
plt.title('Natural Gas Price Model — Fit + Forecast')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.show()