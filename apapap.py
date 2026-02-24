#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib

# --- Load trained model ---
model = joblib.load("sales_forecast_model.pkl")

st.title("Amazon Sales Forecasting App (Simplified)")

st.write("Enter order details to predict Net Sales")

# --- User input form ---
qty = st.number_input("Quantity", min_value=1, value=3)
price = st.number_input("Unit Price", min_value=1.0, value=25000.0)
disc = st.number_input("Discount", min_value=0.0, value=100.0)
ship = st.number_input("Shipping Cost", min_value=0.0, value=80.0)

# --- Dropdown for Category (3 options) ---
category_options = ["Electronics", "Clothing", "Home"]
cat = st.selectbox("Category", category_options)

# --- Dropdown for Sub-Category (7 options) ---
subcat_options = ["Mobile", "Laptop", "Clothing", "Home", "Books", "Toys", "Accessories"]
subcat = st.selectbox("Sub-Category", subcat_options)

# --- Dropdown for Payment Method (4 options) ---
payment_options = ["Credit Card", "UPI", "COD", "Debit Card"]
pay = st.selectbox("Payment Method", payment_options)

# --- Auto-fill defaults ---
location = "USA"
delivery_time_days = 5

# --- Feature engineering ---
gross_sales = qty * price
net_sales = gross_sales - disc - ship
profit_margin_pct = (net_sales / gross_sales) * 100 if gross_sales != 0 else 0

# --- Build input DataFrame ---
input_data = pd.DataFrame([[qty, price, disc, ship, gross_sales, profit_margin_pct,
                            delivery_time_days, location, cat, subcat, pay]],
                          columns=['quantity','unit_price','discount','shipping_cost',
                                   'gross_sales','profit_margin_pct','delivery_time_days',
                                   'customer_location','category','sub_category','payment_method'])

# --- Prediction ---
if st.button("Predict Net Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Net Sales: {round(prediction, 2)}")


# In[ ]:




