#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance - RUL", layout="centered")

st.title("🛠️ Predictive Maintenance: Remaining Useful Life Estimator")
st.markdown("This app uses sensor data to predict Remaining Useful Life (RUL) of jet engines using ML.")

st.markdown("""
### 📄 Project Case Study
**Project Title:**  
**Predictive Maintenance: Remaining Useful Life (RUL) Estimator for Jet Engines**

**Overview:**  
This project demonstrates a machine learning-based predictive maintenance system that estimates the Remaining Useful Life (RUL) of jet engines using NASA’s CMAPSS dataset. The tool was built as an interactive Streamlit app, allowing users to test different models and simulate predictive alerts for early maintenance.

**Problem Statement:**  
In industries like aviation, engine failures can cause massive downtime, safety risks, and financial loss. Predicting when a machine is likely to fail — before it does — is critical. This project aims to address that by forecasting how many cycles an engine has left before maintenance is needed.

**Approach & Tools Used:**
- **Dataset:** NASA CMAPSS FD001 (Sensor readings from multiple engines over time)
- **Features:** Selected 14 key sensors relevant to engine degradation patterns
- **Modeling:**  
  - Linear Regression (baseline model)  
  - XGBoost Regressor (optimized tree-based model for higher accuracy)
- **Frontend:** Streamlit
- **Libraries:** `pandas`, `scikit-learn`, `xgboost`, `matplotlib`
- **Key Features:**
  - Upload your own sensor data (`.txt` or `.csv`)
  - Choose between regression models
  - Visualize actual vs predicted RUL
  - Receive alerts for engines predicted to fail soon
  - Export predictions as a CSV

**Results:**
- Achieved R² scores ranging from 0.65–0.89 depending on model selection and feature tuning
- Built a modular pipeline suitable for integration into larger industrial IoT systems
- Enabled non-technical users to interact with and test models through a no-code interface

**Outcome:**  
This project showcases my skills in applied machine learning, data pipeline engineering, and UI development for industrial use cases. It demonstrates how predictive maintenance can be implemented and evaluated in real-world settings, with real-time alerting and deployable interfaces.

**GitHub Repository:**  
🔗 *[your GitHub link here]*

**Live App:**  
🌐 *[Link to deployed Streamlit app — coming soon]*
""")

# Upload file or use default
txt_file = st.file_uploader("📁 Upload engine data file", type=["txt", "csv"])
if txt_file is not None:
    raw_data = pd.read_csv(txt_file, sep=" ", header=None)
    st.success("✅ File uploaded successfully!")
else:
    st.info("ℹ️ Using default: test_FD001.txt")
    raw_data = pd.read_csv("test_FD001.txt", sep=" ", header=None)

# Load Data
@st.cache_data
def load_data(raw_data):
    columns = [
        "unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3",
        "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
        "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
        "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
        "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
    ]
    raw_data.drop([26, 27], axis=1, inplace=True)
    raw_data.columns = columns

    rul_df = raw_data.groupby("unit_number")["time_in_cycles"].max().reset_index()
    rul_df.columns = ["unit_number", "max_cycle"]
    raw_data = raw_data.merge(rul_df, on="unit_number", how="left")
    raw_data["RUL"] = raw_data["max_cycle"] - raw_data["time_in_cycles"]
    raw_data.drop("max_cycle", axis=1, inplace=True)

    selected_cols = [
        "unit_number", "time_in_cycles", "sensor_2", "sensor_3", "sensor_4",
        "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12",
        "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21", "RUL"
    ]
    raw_data = raw_data[selected_cols]
    return raw_data

df = load_data(raw_data)

# Proper engine-wise train-test split
unique_ids = df["unit_number"].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
train_df = df[df["unit_number"].isin(train_ids)]
test_df = df[df["unit_number"].isin(test_ids)]

features = [col for col in df.columns if col not in ["unit_number", "time_in_cycles", "RUL"]]
X_train = train_df[features]
y_train = train_df["RUL"]
X_test = test_df[features]
y_test = test_df["RUL"]

if X_test.empty:
    st.error("❌ Error: Test set is empty. Please check dataset and splitting logic.")
    st.stop()

# Model selection
model_choice = st.selectbox("🤖 Choose your model", ["Linear Regression", "XGBoost"])
if model_choice == "Linear Regression":
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
else:
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=100, random_state=42)

# Train Model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.markdown(f"- **MAE**: {mae:.2f}")
st.markdown(f"- **RMSE**: {rmse:.2f}")
st.markdown(f"- **R² Score**: {r2:.2f}")

# Plot Results
st.subheader("📈 Actual vs Predicted RUL (First 100 Samples)")
fig, ax = plt.subplots()
ax.scatter(y_test[:100], y_pred[:100], alpha=0.6, label="Predictions", color="blue")
ax.plot([0, max(y_test[:100])], [0, max(y_test[:100])], linestyle="--", color="red", label="Ideal")
ax.set_xlabel("Actual RUL")
ax.set_ylabel("Predicted RUL")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Alert System
st.subheader("🚨 Failure Risk Alerts")
threshold = st.slider("Alert Threshold (RUL in cycles)", 5, 50, 20)
test_df = test_df.copy()
test_df["predicted_RUL"] = y_pred
risky_engines = test_df[test_df["predicted_RUL"] < threshold]["unit_number"].unique()

if len(risky_engines) > 0:
    st.error(f"⚠️ Engines predicted to fail soon: {', '.join(map(str, risky_engines))}")
else:
    st.success("✅ No engines predicted to fail soon.")

# Download predictions
st.subheader("⬇️ Export Predictions")
download_df = test_df[["unit_number", "time_in_cycles", "predicted_RUL"]]
csv = download_df.to_csv(index=False)
st.download_button("Download Predictions as CSV", csv, "predicted_rul.csv", "text/csv")





