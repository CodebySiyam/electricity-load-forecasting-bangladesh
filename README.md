# Electricity Load Forecasting for Bangladesh
## 📌 Project Overview
This project develops accurate short-term electricity demand forecasting models for Bangladesh by combining historical power system data from the Power Grid Company of Bangladesh (PGCB) with relevant weather variables.
Accurate load forecasting is critical for efficient energy planning, optimizing generation schedules, minimizing load shedding, and supporting the integration of renewable energy sources in Bangladesh’s rapidly growing power sector.
Multiple forecasting approaches were evaluated — including tree-based machine learning, deep learning, and statistical methods — to identify the most reliable solution for real-world deployment.

## 🎯 Objectives
- Build high-accuracy models to predict hourly or daily electricity demand
- Quantify the impact of weather and operational variables on power consumption
- Compare traditional statistical, machine learning, and deep learning techniques
- Develop an end-to-end production-ready forecasting pipeline
- Create an interactive web application for stakeholders

## 📊 Dataset 

**Primary Source:** Hourly generation, demand, and load-shedding records from PGCB (Power Grid Company of Bangladesh).
**Supplementary Data:** Historical weather information (temperature, humidity, rainfall, wind speed, etc.) for major load centers.

**Key Variables:**
  - Electricity demand and generation (MW)
  - Load shedding (MW)
  - Energy mix (Gas, Coal, Hydro, Solar, Wind, Imports)
  - Weather: Temperature, Humidity, Precipitation, Wind Speed
  - Temporal features: Hour, Day, Weekday, Month, Holiday/Weekend flags
**Data Pipeline:** PGCB data was collected, cleaned, and merged with weather data using datetime alignment. Missing values were handled through interpolation and forward/backward filling where appropriate.

## ⚙️ Methodology & Feature Engineering

- Data Collection & Integration: Automated scripts for PGCB data ingestion + weather API/station data merging
- Time Series Decomposition: Applied STL (Seasonal-Trend decomposition using LOESS) to separate trend, seasonal, and residual components. These decomposed features were used as additional inputs to improve model understanding of underlying patterns.
- Feature Engineering:
   - Lag features and rolling statistics for temporal dependencies.
   - Cyclical encoding for hour, day, and month.
   - Categorical encoding for energy sources and holiday flags
   - Interaction terms between temperature and humidity
   - Outlier detection and Handling(Extreme Anomalies).
- Train-Test Split: Time-based splitting to prevent data leakage

## Models Evaluated
- **XGBoost (Machine Learning):** Gradient boosting framework excellent at capturing non-linear relationships in tabular data.
- **LSTM (Deep Learning):** Long Short-Term Memory networks for sequential pattern learning.
- **Prophet (Statistical):** Facebook’s additive model focused on trend, seasonality, and holiday effects.

## 📈 Model Performance Comparison

| Model       | Category            | MAE (↓)   | RMSE (↓)    | R² (↑)   | MAPE (↓) |
|-------------|---------------------|-----------|-------------|----------|----------|
| **XGBoost** | Machine Learning    | **194.59**| **1099.41** | **0.8020**| **1.81%** |
| LSTM        | Deep Learning       | 551.28    | 1232.60     | 0.7511   | 4.96%    |
| Prophet     | Statistical         | 913.26    | 1552.84     | 0.6050   | 8.71%    |

### 🏆 Best Model: XGBoost

**Why XGBoost Outperformed:**

- Superior handling of structured/tabular data with mixed feature types
- Excellent at modeling complex non-linear interactions (especially weather + temporal features)
- More robust to noise and outliers compared to neural networks on this dataset
- Significantly faster training and inference
- Better interpretability through feature importance

## 📊 Key Insights

- Weather variables (particularly temperature and humidity) are among the strongest predictors of electricity demand
- STL decomposition effectively revealed strong daily and seasonal patterns in Bangladesh’s load profile
- Machine learning approaches significantly outperformed classical statistical and deep learning models for this dataset
- Feature engineering and proper temporal alignment were critical to achieving sub-2% MAPE
- Load shedding events show distinct patterns that can be anticipated with improved forecasting

## 🖥️ Interactive Web Application (Streamlit)

A user-friendly Streamlit dashboard was developed for real-time forecasting.

**Features:**
- Intuitive input form for key variables
- Instant demand prediction using the XGBoost model
- Interactive visualizations (actual vs predicted, feature importance, STL components)
- Historical trend explorer
- Exportable reports

## 🚀 Future Enhancements

- Cloud deployment (AWS / Streamlit Community Cloud)
- Real-time data pipeline with live PGCB and weather feeds
- Ensemble models combining strengths of XGBoost and LSTM
- 24–72 hour ahead rolling forecasts
- API development for integration with utility systems
- Uncertainty quantification and probabilistic forecasting

## 🛠️ Tech Stack

- **Language**: Python
- **Data Handling**: Pandas, NumPy
- **Modeling**: XGBoost, TensorFlow/Keras, Prophet, statsmodels (STL)
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Web App**: Streamlit
- **Other**: Scikit-learn, Joblib, Jupyter

## 👨‍💻 Author

**Md Siyam Hossain**  
Data Science & Machine Learning Enthusiast
