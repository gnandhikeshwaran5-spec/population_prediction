# Updated streamlit_app.py

import streamlit as st
import numpy as np

# Sample model prediction array
prediction = np.array([[1.5]])  # Example prediction

# Correcting TypeError by extracting the scalar value
predicted_value = float(prediction[0][0])

st.title('Population Prediction App')
st.write('Predicted Population:', predicted_value)