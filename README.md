# JPMorgan Commodity Forecasting Model

An interactive Streamlit app that forecasts natural gas prices using Python, pandas, NumPy, SciPy, Matplotlib, and Streamlit.

## Live Demo

[Open Live App](https://osa-jpm-commodity-forecast.streamlit.app/)

## GitHub Repository

[View Source Code](https://github.com/osa23-rep/jpmorgan-qr-forage)

## Overview

This project was built from a JPMorgan Quantitative Research virtual experience task. The model uses historical natural gas price data and fits a trend-plus-seasonality regression model to estimate future commodity prices.

## Features

- Analyzes historical natural gas price data
- Fits a seasonal regression model with a linear trend and sine-wave component
- Captures 12-month price cycles
- Generates forward price estimates
- Displays model parameters, forecast outputs, and visualizations in Streamlit

## Tools Used

- Python
- pandas
- NumPy
- SciPy
- Matplotlib
- Streamlit

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run gas_price.py
