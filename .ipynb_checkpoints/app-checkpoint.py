import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_pipeline.pkl")
st.set_page_config(
    page_title="Electricity Forecasting",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #00ffcc;
        text-align: center;
        font-size: 40px;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00ccaa;
        color: white;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚡ Bangladesh Electricity Load Forecasting</h1>", unsafe_allow_html=True)

st.write("### Predict electricity demand using Machine Learning (XGBoost Pipeline)")

feature_names = [
    "gen_total_mw",
    "load_shedding_mw",
    "temp_c",
    "humidity_pct",
    "rain_mm",
    "wind_speed_ms",
    "remarks",
    "month_name",
    "day_name",
    "weekly_holiday",
    "weather_condition",
    "import_electricity_mw",
    "total_generation_sources_mw"
]

st.sidebar.header("🔧 Input Features")

input_data = {}

for col in feature_names:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

st.markdown("## Prediction Panel")

col1, col2 = st.columns(2)

with col1:
    st.info("Click the button to predict electricity demand")

with col2:
    if st.button("Predict Demand"):
        prediction = model.predict(input_df)
        st.success(f"Predicted Demand: {prediction[0]:.2f} MW")

with st.expander("Show Input Data"):
    st.dataframe(input_df)