#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from plotly import graph_objects as go
from plotly.subplots import make_subplots


# In[2]:


# Load the dataset
def load_data():
    df = pd.read_csv('pricesforstreamlit.csv', parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    df['Quarter'] = df.index.quarter  # Add Quarter column for filtering
    return df

# Filter Data Function
def filter_data(data, years, quarters, months):
    if not months:  # If no months are selected, include all months in the selected quarters
        months = data[data['Quarter'].isin(quarters)].index.month.unique()
    filtered = data[
        (data.index.year.isin(years)) &
        (data['Quarter'].isin(quarters)) &
        (data.index.month.isin(months))
    ]
    return filtered

# Forecast Prices Function
def forecast_prices(train, test, horizon):
    history = list(train)
    predictions = []

    # Rolling forecast for test data
    for t in range(len(test)):
        model = ARIMA(history, order=(3, 0, 3))
        model_fit = model.fit()
        pred = model_fit.forecast(steps=1)[0]
        predictions.append(pred)
        history.append(test.iloc[t])

    # Forecast future data
    future_forecasts = []
    future_history = list(train) + list(test)
    for _ in range(horizon):
        model = ARIMA(future_history, order=(3, 0, 3))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]
        future_forecasts.append(forecast)
        future_history.append(forecast)

    return predictions, future_forecasts

# Load Dataset
df = load_data()

# Streamlit Configuration
st.set_page_config(page_title="Chicken Price Forecasting", layout="wide")

# Title
st.title("Chicken Price Forecasting")

# Sidebar Filters
st.sidebar.header("Filters")
year_filter = st.sidebar.multiselect("Select Year", options=sorted(df.index.year.unique()), default=sorted(df.index.year.unique()))
quarter_filter = st.sidebar.multiselect("Select Quarter", options=[1, 2, 3, 4], default=[1, 2, 3, 4])
month_filter = st.sidebar.multiselect("Select Month (Optional)", options=range(1, 13), default=[])

# Filter Data
filtered_data = filter_data(df, year_filter, quarter_filter, month_filter)

# Historical Candlestick Chart
st.subheader("Historical Price Trend")
if not filtered_data.empty:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=filtered_data.index,
            open=filtered_data['Actual Price per kg'],
            high=filtered_data['Actual Price per kg'] * 1.05,  # Example: +5%
            low=filtered_data['Actual Price per kg'] * 0.95,   # Example: -5%
            close=filtered_data['Actual Price per kg'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name="Price per kg",
        )
    )

    fig.update_layout(
        title="Historical Chicken Prices (Candlestick)",
        xaxis_title="Date",
        yaxis_title="Price per kg",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig)
else:
    st.warning("No data available for the selected filters.")

# Forecast Section
st.sidebar.header("Forecast Settings")
forecast_horizon = st.sidebar.slider("Select Forecast Horizon (Days):", 30, 180, 60)

if st.sidebar.button("Run Forecast"):
    # Train-Test Split
    train_size = int(len(df) * 0.8)
    train_data = df['Actual Price per kg'][:train_size]
    test_data = df['Actual Price per kg'][train_size:]

    # Forecasting
    st.subheader("Running Forecasting Model...")
    predictions, future_forecasts = forecast_prices(train_data, test_data, forecast_horizon)

    # Generate Future Dates
    future_dates = pd.date_range(start=df.index[-1], periods=forecast_horizon + 1, freq='D')[1:]

    # Evaluate Performance
    mse = mean_squared_error(test_data, predictions)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")

    # Plot Forecasting Results
    st.subheader("Actual vs Predicted Prices with Forecasts")
    fig = go.Figure()

    # Add Training Data
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Training Data', line=dict(color='blue')))

    # Add Test Data
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data, mode='lines', name='Test Data', line=dict(color='orange')))

    # Add Predictions
    fig.add_trace(go.Scatter(x=test_data.index, y=predictions, mode='lines', name='Test Predictions', line=dict(color='green')))

    # Add Future Forecasts
    fig.add_trace(go.Scatter(x=future_dates, y=future_forecasts, mode='lines', name=f'Future {forecast_horizon}-Day Forecast', line=dict(color='red')))

    fig.update_layout(
        title="Actual vs Rolling Forecasted Chicken Prices (Including Future Forecasts)",
        xaxis_title="Date",
        yaxis_title="Price per kg",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig)

    # Display Projected Prices Table
    st.subheader("Projected Prices for Next Days")
    projected_prices = pd.DataFrame({"Date": future_dates, "Projected Price": future_forecasts})
    st.dataframe(projected_prices.style.format({"Projected Price": "{:.2f}"}))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




