#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance - RUL", layout="centered")

st.title("🛠️ Predictive Maintenance: Remaining Useful Life Estimator")
st.markdown("This app uses sensor data to predict Remaining Useful Life (RUL) of jet engines.")

# Load Data
@st.cache_data
def load_data():
    columns = [
        "unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3",
        "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
        "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
        "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
        "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
    ]
    df = pd.read_csv("test_FD001.txt", sep=" ", header=None)
    df.drop([26, 27], axis=1, inplace=True)
    df.columns = columns

    rul_df = df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    rul_df.columns = ["unit_number", "max_cycle"]
    df = df.merge(rul_df, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    df.drop("max_cycle", axis=1, inplace=True)

    selected_cols = [
        "unit_number", "time_in_cycles", "sensor_2", "sensor_3", "sensor_4",
        "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12",
        "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21", "RUL"
    ]
    df = df[selected_cols]
    return df

df = load_data()

# Train/Test Split
train_ids = df["unit_number"].sample(frac=0.8, random_state=42).unique()
test_ids = df[~df["unit_number"].isin(train_ids)]["unit_number"].unique()

train_df = df[df["unit_number"].isin(train_ids)]
test_df = df[df["unit_number"].isin(test_ids)]

features = [col for col in df.columns if col not in ["unit_number", "time_in_cycles", "RUL"]]
X_train = train_df[features]
y_train = train_df["RUL"]
X_test = test_df[features]
y_test = test_df["RUL"]

# Train Model
model = LinearRegression()
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

# Plot: Actual vs Predicted
st.subheader("📈 Actual vs Predicted RUL")

fig, ax = plt.subplots()
ax.scatter(y_test[:100], y_pred[:100], alpha=0.6, color='blue', label="Predictions")
ax.plot([0, max(y_test[:100])], [0, max(y_test[:100])], color='red', linestyle='--', label='Ideal')
ax.set_xlabel("Actual RUL")
ax.set_ylabel("Predicted RUL")
ax.set_title("Actual vs Predicted RUL")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Alert System
st.subheader("🚨 Alert System")
threshold = st.slider("Critical RUL Threshold", min_value=5, max_value=50, value=20)
test_df = test_df.copy()
test_df["predicted_RUL"] = model.predict(X_test)
risky = test_df[test_df["predicted_RUL"] < threshold]["unit_number"].unique()

st.markdown(f"**Engines predicted to fail soon (RUL < {threshold}):**")
st.code(risky if len(risky) > 0 else "No engines below threshold.")


# In[ ]:




