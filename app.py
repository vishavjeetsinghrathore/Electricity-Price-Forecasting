# app.py

import streamlit as st
import pandas as pd
import joblib
import os
import subprocess

st.set_page_config(page_title="Electricity Load Forecasting")

# ===============================
# Train model if not exists
# ===============================
MODEL_PATH = "best_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training model... ⏳")
    subprocess.run(["python", "train_model.py"])

# ===============================
# Load trained model
# ===============================
model = joblib.load(MODEL_PATH)

st.title("⚡ Electricity Load Forecasting")
st.write("Predict next hour electricity consumption (MW)")

# ===============================
# User Inputs
# ===============================
hour = st.slider("Hour", 0, 23, 12)
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 2)

lag1 = st.number_input("Previous Hour Load", value=13000.0)
lag2 = st.number_input("Load 2 Hours Ago", value=12800.0)
lag3 = st.number_input("Load 3 Hours Ago", value=12700.0)

# ===============================
# Prediction
# ===============================
if st.button("Predict Load"):
    input_df = pd.DataFrame({
        "hour": [hour],
        "day": [day],
        "month": [month],
        "dayofweek": [dayofweek],
        "lag_1": [lag1],
        "lag_2": [lag2],
        "lag_3": [lag3]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"🔮 Predicted Electricity Load: {prediction:.2f} MW")
