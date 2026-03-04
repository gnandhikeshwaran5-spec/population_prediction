import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="India Population AI", layout="wide")

st.title("🇮🇳 India Population Prediction")
st.markdown("### AI powered population forecasting system")

st.image("https://images.unsplash.com/photo-1524492412937-b28074a5d7da", use_column_width=True)

st.write("---")

# MODEL
years = np.array([2000,2005,2010,2015,2020]).reshape(-1,1)
population = np.array([1056,1147,1234,1311,1380])

model = LinearRegression()
model.fit(years, population)

# INPUT
st.subheader("Select Future Year")

year = st.slider("Year", 2025, 2100, 2030)

prediction = model.predict([[year]])

# RESULT CARDS
col1, col2 = st.columns(2)

with col1:
    st.metric("Selected Year", year)

with col2:
    st.metric("Predicted Population (Millions)", round(prediction[0],2))

st.write("---")

# INFORMATION SECTION
st.subheader("About Population Growth")

st.write("""
India is currently the most populous country in the world.
Population growth impacts economy, infrastructure, and resources.

Machine learning models can help forecast population trends
based on historical data.
""")

st.write("---")

# VIDEO SECTION
st.subheader("Understanding Population Growth")

st.video("https://www.youtube.com/watch?v=PUwmA3Q0_OE")

st.success("Prediction generated using Linear Regression.")
