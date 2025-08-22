import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tempfile
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# -------------------------------
# Sidebar Navigation
# -------------------------------
PAGES = ["Introduction", "EDA", "Training", "Evaluation", "Prediction"]
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", PAGES)

# -------------------------------
# Helper Functions
# -------------------------------
def load_image(img_file):
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def build_model(model_name, input_shape=(224,224,3)):
    if model_name == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "VGG16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    preds = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=base_model.input, outputs=preds)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -------------------------------
# Dataset Upload Section (common)
# -------------------------------
def handle_dataset_upload():
    uploaded_file = st.file_uploader("Upload dataset (zip file with class subfolders)", type=["zip"])
    if uploaded_file:
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        st.success("Dataset extracted successfully!")
        return tmp_dir
    return None

# -------------------------------
# Pages
# -------------------------------
if page == "Introduction":
    st.title("Tuberculosis Detection Using Deep Learning")
    st.write("""
    This app allows you to:
    - Upload dataset as a zip file
    - Explore the dataset (EDA)
    - Train deep learning models (ResNet50, VGG16, EfficientNetB0)
    - Evaluate models with metrics and plots
    - Upload chest X-ray images for TB prediction
    """)

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    dataset_path = handle_dataset_upload()
    if dataset_path:
        classes = os.listdir(dataset_path)
        st.write("Classes found:", classes)
        
        img_paths = []
        labels = []
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            if os.path.isdir(c_dir):
                files = os.listdir(c_dir)[:5]
                for f in files:
                    img_paths.append(os.path.join(c_dir, f))
                    labels.append(c)
        
        if img_paths:
            df = pd.DataFrame({"image": img_paths, "label": labels})
            st.write(df.head())
            
            fig, ax = plt.subplots()
            sns.countplot(x="label", data=df)
            st.pyplot(fig)
            
            st.image(df["image"].iloc[0], caption=df["label"].iloc[0])
            st.session_state["dataset_path"] = dataset_path
        else:
            st.warning("No images found inside the dataset.")
    else:
        st.info("Upload a dataset zip file to proceed.")

elif page == "Training":
    st.title("Model Training")
    dataset_path = st.session_state.get("dataset_path", None)
    model_choice = st.selectbox("Choose Model", ["ResNet50", "VGG16", "EfficientNetB0"])
    epochs = st.slider("Epochs", 1, 20, 5)
    batch_size = st.slider("Batch Size", 8, 64, 16)
    
    if st.button("Start Training"):
        if dataset_path and os.path.exists(dataset_path):
            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
            train_gen = datagen.flow_from_directory(dataset_path, target_size=(224,224),
                                                    batch_size=batch_size, class_mode="binary", subset="training")
            val_gen = datagen.flow_from_directory(dataset_path, target_size=(224,224),
                                                  batch_size=batch_size, class_mode="binary", subset="validation")
            
            model = build_model(model_choice)
            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
            
            st.session_state["model"] = model
            st.session_state["history"] = history.history
            st.session_state["val_gen"] = val_gen
            st.success("Training complete!")
            
            # Plot training curve
            fig, ax = plt.subplots()
            ax.plot(history.history['accuracy'], label='train acc')
            ax.plot(history.history['val_accuracy'], label='val acc')
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Please upload and explore dataset first.")

elif page == "Evaluation":
    st.title("Model Evaluation")
    if "model" in st.session_state and "val_gen" in st.session_state:
        model = st.session_state["model"]
        val_gen = st.session_state["val_gen"]
        
        # Predictions
        y_true = val_gen.classes
        y_pred_probs = model.predict(val_gen)
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        # Classification Report
        report = classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()), output_dict=True)
        st.write(pd.DataFrame(report).transpose())
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_gen.class_indices.keys(), yticklabels=val_gen.class_indices.keys())
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)
        
        # ROC Curve
        if len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_pred_probs)
            fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0,1],[0,1],'--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("Please train a model first.")

elif page == "Prediction":
    st.title("TB Prediction")
    uploaded_file = st.file_uploader("Upload a Chest X-ray", type=["jpg", "png", "jpeg"])
    if uploaded_file and "model" in st.session_state:
        img = load_image(uploaded_file)
        st.image(img, caption="Uploaded Image")
        
        img_resized = cv2.resize(img, (224,224)) / 255.0
        pred = st.session_state["model"].predict(np.expand_dims(img_resized, axis=0))[0][0]
        label = "Tuberculosis" if pred > 0.5 else "Normal"
        st.success(f"Prediction: {label} (score: {pred:.2f})")
    elif uploaded_file:
        st.warning("Train a model first before prediction.")
