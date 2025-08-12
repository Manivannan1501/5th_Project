import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image, ImageOps
import os
import tempfile
import datetime

# Configure page
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Dependency Check ----
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.sidebar.warning("OpenCV not installed - advanced image processing disabled")

# ---- Custom CSS ----
st.markdown("""
<style>
    .css-1vq4p4l {padding: 2rem 1rem;}
    .st-emotion-cache-1y4p8pa {width: 100%; padding: 2rem 1rem;}
    .header {color: #2a9d8f; border-bottom: 1px solid #eee; padding-bottom: 0.5rem;}
    .result-box {border-radius: 10px; padding: 1.5rem; margin: 1rem 0;}
    .tb-positive {background-color: #ffebee; border-left: 5px solid #f44336;}
    .tb-negative {background-color: #e8f5e9; border-left: 5px solid #4caf50;}
    .confidence-bar {height: 20px; border-radius: 10px; background: #f5f5f5; margin: 0.5rem 0;}
    .confidence-fill {height: 100%; border-radius: 10px; background: #1e88e5;}
</style>
""", unsafe_allow_html=True)

# ---- Model Loading ----
@st.cache_resource
def load_model():
    """Load the trained TB detection model"""
    model_path = 'tb_detection_model.h5'
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
        
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# ---- Image Processing ----
def preprocess_image(file):
    """Process uploaded image for model prediction"""
    try:
        img = Image.open(file)
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize and normalize
        img = ImageOps.fit(img, (224, 224), Image.LANCZOS)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img, img_array
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None, None

# ---- Main App ----
def main():
    st.title("Tuberculosis Detection from Chest X-Rays")
    st.markdown("""
    <p style="color: #666; font-size: 1.1rem;">
    AI-powered screening tool for detecting signs of tuberculosis in chest radiographs.
    For research use only - not for clinical diagnosis.
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        - Upload a chest X-ray image (JPEG/PNG)
        - System will analyze for TB indicators
        - Results show confidence percentage
        - See GitHub: [TB Detection Project](https://github.com/your-repo)
        """)
        
        st.header("Model Info")
        st.write("Deep CNN trained on 3,000+ X-ray images")
        if 'model' in st.session_state:
            st.success("Model loaded successfully!")
        
        st.header("Limitations")
        st.warning("Not a substitute for professional medical diagnosis")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Select a high-quality frontal chest radiograph"
    )

    # Display and analyze image
    if uploaded_file is not None:
        # Create layout columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, caption=f"{uploaded_file.name} ({img.size[0]}×{img.size[1]})", use_column_width=True)
        
        # Process when analyze button clicked
        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Processing image..."):
                model = load_model()
                if model is None:
                    st.error("Cannot proceed without model")
                    st.stop()
                
                # Process image
                img, img_array = preprocess_image(uploaded_file)
                if img_array is None:
                    st.error("Failed to process image")
                    st.stop()
                
                # Make prediction
                try:
                    prediction = model.predict(img_array)
                    tb_prob = float(prediction[0][0])
                    confidence = tb_prob if tb_prob > 0.5 else 1 - tb_prob
                    prediction_class = "Tuberculosis Detected" if tb_prob > 0.5 else "Normal"
                    
                    # Display results
                    with col2:
                        st.subheader("Analysis Results")
                        
                        # Result box
                        result_class = "tb-positive" if tb_prob > 0.5 else "tb-negative"
                        st.markdown(f"""
                        <div class="result-box {result_class}">
                            <h3 style="margin-top: 0;">{prediction_class}</h3>
                            <p>Confidence: <strong>{confidence*100:.1f}%</strong></p>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence*100}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Clinical notes
                        if tb_prob > 0.5:
                            st.warning("""
                            **Clinical Note:**  
                            This result suggests possible TB infection. 
                            Please consult a pulmonologist immediately for confirmatory tests.
                            """)
                        else:
                            st.info("""
                            **Clinical Note:**  
                            No signs of active TB detected. Routine follow-up recommended 
                            if patient has risk factors or persistent symptoms.
                            """)
                            
                        # Add debug info expander
                        with st.expander("Technical Details"):
                            st.write(f"Prediction score: {tb_prob:.4f}")
                            st.write(f"Model threshold: >0.5 for TB")
                            st.write(f"Processing time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    # Footer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This AI tool is intended for research and educational purposes only.  
    Not intended for clinical use or medical decision-making. Developed by Medical AI Research Group.
    """)

if __name__ == "__main__":
    main()
