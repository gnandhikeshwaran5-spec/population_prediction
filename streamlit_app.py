import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("India Population Prediction")

st.write("Predict India's population using Machine Learning")

# sample dataset
years = np.array([2000,2005,2010,2015,2020]).reshape(-1,1)
population = np.array([1056,1147,1234,1311,1380])

model = LinearRegression()
model.fit(years, population)

year = st.number_input("Enter Year", min_value=2025, max_value=2100)

prediction = model.predict([[year]])

st.subheader("Predicted Population (Millions)")
st.write(float(prediction))
