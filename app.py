import streamlit as st
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("model/iris_model.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

# App title
st.title("🌸 Iris Flower Classification App")

st.write("Enter the flower measurements below:")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict Species"):
    # Prepare input for model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    species = label_encoder.inverse_transform(prediction)[0]
    
    # Show result
    st.success(f"🌼 Predicted Iris Species: **{species}**")
