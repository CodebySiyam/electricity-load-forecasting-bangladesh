import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("xgb_pipeline.pkl")

st.set_page_config(
    page_title="Electricity Forecasting",
    layout="wide"
)

st.markdown("<h1>⚡ Bangladesh Electricity Load Forecasting</h1>", unsafe_allow_html=True)

st.write("### Predict electricity demand")

num_features = [
    "gen_total_mw",
    "load_shedding_mw",
    "temp_c",
    "humidity_pct",
    "rain_mm",
    "wind_speed_ms",
    "import_electricity_mw",
    "total_generation_sources_mw"
]

st.sidebar.header("Input Features")

input_data = {}

for col in num_features:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_data["day_name"] = st.sidebar.selectbox(
    "day_name",
    ["Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday"]
)

input_data["month_name"] = st.sidebar.selectbox(
    "month_name",
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"]
)

input_data["weekly_holiday"] = st.sidebar.selectbox(
    "weekly_holiday",
    ["Yes", "No"]
)

input_data["weather_condition"] = st.sidebar.selectbox(
    "weather_condition",
    ["cold", "heatwave", "normal"]
)

input_data["remarks"] = st.sidebar.selectbox(
    "remarks",
    ["Evening_Peak", "Day_Peak", "Night_OffPeak", "Late_Evening"]
)


input_df = pd.DataFrame([input_data])


st.markdown("## Prediction Panel")

col1, col2 = st.columns(2)

with col1:
    st.info("Click button to predict electricity demand")

with col2:

    if st.button("Predict Demand"):
        
        # Prediction
        prediction = model.predict(input_df)[0]

        st.success(f"🔥 Predicted Demand: {prediction:.2f} MW")



if st.button("📈 Predict Next 24 Hours"):

    predictions = []

    for i in range(24):
        pred = model.predict(input_df)[0]
        predictions.append(pred)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(range(24), predictions, marker="o")
    ax.set_title("Next 24 Hour Electricity Forecast")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")

    st.pyplot(fig)


with st.expander("📌 Show Input Data"):
    st.dataframe(input_df)