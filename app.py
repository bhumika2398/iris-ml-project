import streamlit as st
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification App")

st.markdown(
    """
This app predicts the **species of an Iris flower** based on its measurements.

**How to use:**
1. Enter the flower measurements in centimeters
2. Click **Predict**
3. View the predicted species
"""
)

import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("model/iris_model.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

# App title
st.title("🌸 Iris Flower Classification App")

st.write("Enter the flower measurements below:")

# Input fields
st.subheader("🌼 Enter Flower Measurements (cm)")

sepal_length = st.number_input(
    "Sepal Length (cm)", min_value=0.0, value=5.1
)
sepal_width = st.number_input(
    "Sepal Width (cm)", min_value=0.0, value=3.5
)
petal_length = st.number_input(
    "Petal Length (cm)", min_value=0.0, value=1.4
)
petal_width = st.number_input(
    "Petal Width (cm)", min_value=0.0, value=0.2
)


# Predict button
st.markdown("---")

if st.button("🔮 Predict Species"):
    features = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]
    
    prediction = model.predict(features)
    species = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"🌸 **Predicted Iris Species:** {species}")

st.markdown(
    """
---
📌 **Project:** Iris Flower Classification  
🛠️ **Built with:** Python, scikit-learn, Streamlit  
📦 **Deployed on:** Streamlit Community Cloud
"""
)
