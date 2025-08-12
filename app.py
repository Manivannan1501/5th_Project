import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import os
import tempfile

# Check for optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not available - some visualization features will be limited")

# Page configuration
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        # Load model from local file or URL
        model_path = 'tb_detection_model.h5'
        
        if not os.path.exists(model_path):
            st.error("Model file not found!")
            return None
            
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
        
        # Using PIL instead of cv2 for basic operations
        img = Image.open(tmp_file.name)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        os.unlink(tmp_file.name)
        return img, img_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    st.title("Tuberculosis Detection System")
    
    model = load_model()
    if model is None:
        st.stop()
    
    uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
        if st.button('Analyze'):
            img, processed_img = preprocess_image(uploaded_file)
            
            if processed_img is not None:
                prediction = model.predict(processed_img)
                confidence = float(np.max(prediction))
                
                with col2:
                    st.subheader("Results")
                    
                    if prediction[0][0] > 0.5:
                        st.error(f"Tuberculosis Detected (Confidence: {confidence*100:.1f}%)")
                        st.warning("Please consult a medical professional immediately")
                    else:
                        st.success(f"Normal (Confidence: {(1-confidence)*100:.1f}%)")
                        st.info("No signs of tuberculosis detected")
                    
                    st.write("Note: This tool is for preliminary screening only")

if __name__ == '__main__':
    main()
