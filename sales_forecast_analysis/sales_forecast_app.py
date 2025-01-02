import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model (use the correct file path)
model = joblib.load('sales_forecasting_model.pkl')

# Streamlit app UI for input
st.title("Sales Forecasting")
st.write("Enter the features of the product to forecast sales:")

# User input fields
month = st.number_input('Month', min_value=1, max_value=12)
day_of_week = st.number_input('Day of the Week', min_value=0, max_value=6)
lag_1 = st.number_input('Lag 1 (Previous sales)', min_value=0)
rolling_mean_7 = st.number_input('7-day Rolling Mean of Sales', min_value=0)

# Predict sales using the trained model
if st.button('Predict Sales'):
    prediction = model.predict([[month, day_of_week, lag_1, rolling_mean_7]])
    st.write(f"Predicted Sales: {prediction[0]}")
