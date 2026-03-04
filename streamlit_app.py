import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="India Population Prediction", layout="wide")

st.title("🇮🇳 India Population Prediction Dashboard")
st.markdown("Machine Learning based population forecasting system")

# Dataset
years = np.array([2000,2005,2010,2015,2020]).reshape(-1,1)
population = np.array([1056,1147,1234,1311,1380])

model = LinearRegression()
model.fit(years, population)

# Sidebar input
st.sidebar.header("Prediction Settings")
year = st.sidebar.slider("Select Year", 2025, 2100, 2030)

prediction = model.predict([[year]])

# Metrics
col1, col2 = st.columns(2)

with col1:
    st.metric("Selected Year", year)

with col2:
    st.metric("Predicted Population (Millions)", round(prediction[0],2))

# Chart
future_years = np.arange(2000,2101,5).reshape(-1,1)
future_predictions = model.predict(future_years)

df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Population": future_predictions
})

st.subheader("Population Trend")

fig, ax = plt.subplots()
ax.plot(df["Year"], df["Population"])
ax.scatter(years, population)
ax.set_xlabel("Year")
ax.set_ylabel("Population (Millions)")
ax.set_title("India Population Forecast")

st.pyplot(fig)

st.info("This prediction is generated using Linear Regression on historical population data.")
