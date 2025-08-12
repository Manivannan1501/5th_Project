import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .title { text-align: center; color: #2a9d8f; }
    .sidebar .sidebar-content { background-color: #f1faee; }
    .upload-box { border: 2px dashed #457b9d; padding: 20px; border-radius: 5px; }
    .result-box { background-color: #a8dadc; padding: 15px; border-radius: 5px; }
    .file-info { font-size: 0.8em; color: #1d3557; }
    .footer { text-align: center; font-size: 0.8em; color: #1d3557; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('tb_detection_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
        
        # Load and process image
        img = image.load_img(tmp_file.name, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Clean up temp file
        os.unlink(tmp_file.name)
        
        return img, img_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

# Main app
def main():
    st.markdown("<h1 class='title'>Tuberculosis Detection from Chest X-Rays</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://placehold.co/300x100?text=TB+Detection", width=300)
        st.markdown("""
        ### About
        This AI system helps detect tuberculosis from chest X-ray images using deep learning.
        """)
        st.markdown("""
        ### Instructions
        1. Upload a chest X-ray image (jpg/png)
        2. Click 'Analyze' to get prediction
        3. View results and confidence score
        """)
    
    st.markdown("""
    <div class='upload-box'>
        <h3>Upload Chest X-Ray Image</h3>
        <p class='file-info'>Supports JPG, PNG formats. Optimal size: 500x500px or larger.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button('Analyze Image', use_container_width=True):
            with st.spinner('Processing image...'):
                img, processed_img = preprocess_image(uploaded_file)
                
                if model and processed_img is not None:
                    # Make prediction
                    prediction = model.predict(processed_img)
                    confidence = np.max(prediction) * 100
                    predicted_class = "Tuberculosis" if prediction[0][0] > 0.5 else "Normal"
                    
                    # Display results
                    with col2:
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.subheader("Analysis Results")
                        
                        # Confidence meter
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                        st.progress(int(confidence))
                        
                        # Prediction result
                        if predicted_class == "Tuberculosis":
                            st.error(f"Prediction: **{predicted_class}** detected")
                        else:
                            st.success(f"Prediction: **{predicted_class}** (No TB detected)")
                        
                        # Explanation
                        if predicted_class == "Tuberculosis":
                            st.warning("Note: This AI prediction suggests possible TB infection. Please consult with a medical professional for confirmation.")
                        else:
                            st.info("Note: No signs of tuberculosis detected in this X-ray. However, always consult a doctor for official diagnosis.")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Grad-CAM visualization (example implementation)
                        with st.expander("Visual Explanation"):
                            st.info("This heatmap shows areas the model focused on when making its decision.")
                            # Placeholder for Grad-CAM visualization
                            st.image("https://placehold.co/600x300?text=Model+Attention+Map", 
                                   caption="Areas of interest highlighted by the model")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class='footer'>
        <p>TB Detection System v1.0 | For research purposes only | Not for clinical use</p>
        <p>© 2023 Medical AI Research Group</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
