import streamlit as st
import joblib
import pandas as pd
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models", "best_xgboost_model.pkl")
features_path = os.path.join(project_root, "models", "model_features.pkl")

model = joblib.load(model_path)
features = joblib.load(features_path)

st.title("Azure Demand Forecast Dashboard")

region = st.selectbox(
    "Select Region",
    ["east-us", "west-europe", "southeast-asia"]
)

service = st.selectbox(
    "Select Service Type",
    ["compute", "storage"]
)

capacity = st.number_input(
    "Provisioned Capacity",
    100,1000,500
)

availability = st.number_input(
    "Availability %",
    90.0,100.0,99.5
)

economic_index = st.number_input(
    "Economic Index",
    80.0,120.0,100.0
)

market_index = st.number_input(
    "Market Demand Index",
    80.0,120.0,100.0
)

if st.button("Generate Forecast"):

    input_data = pd.DataFrame({
        "provisioned_capacity":[capacity],
        "availability_pct":[availability],
        "economic_index":[economic_index],
        "market_demand_index":[market_index]
    })

    for col in features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[features]

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Usage: {prediction:.2f}")

    utilization = prediction / capacity

    st.write(f"Expected Utilization: {utilization*100:.2f}%")

    if utilization > 0.85:
        st.error("⚠ Capacity may be insufficient. Scale infrastructure.")
    else:
        st.success("Capacity sufficient")