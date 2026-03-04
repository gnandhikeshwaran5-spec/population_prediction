import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="India Population Prediction", layout="wide")

st.title("India Population Prediction")

# Model
years = np.array([2000,2005,2010,2015,2020]).reshape(-1,1)
population = np.array([1056,1147,1234,1311,1380])

model = LinearRegression()
model.fit(years, population)

# Slider input
st.subheader("Select Future Year")
year = st.slider("Year", 2025, 2100, 2030)

prediction = model.predict([[year]])

# Prediction display
col1, col2 = st.columns(2)

with col1:
    st.metric("Selected Year", year)

with col2:
    st.metric("Predicted Population (Millions)", round(prediction[0],2))


st.write("---")

# IMAGE BELOW PREDICTION
st.subheader("Population Data Visualization")

st.image(
"https://images.unsplash.com/photo-1551288049-bebda4e38f71",
use_column_width=True
)
