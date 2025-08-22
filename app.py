import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os

# Load the trained model
model = load_model('model.ipynb')  # Update with your model path

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app layout
st.title("Tuberculosis Detection from Chest X-ray Images")
st.write("Upload a chest X-ray image to check for signs of Tuberculosis.")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(np.array(img))
    
    # Make prediction
    prediction = model.predict(img_array)
    class_labels = ['Normal', 'Tuberculosis']  # Update with your class labels
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display the prediction
    st.write(f"Prediction: {predicted_class} with confidence {confidence:.2f}%")

# Additional information
st.write("This application uses a deep learning model to classify chest X-ray images as either normal or showing signs of tuberculosis.")
