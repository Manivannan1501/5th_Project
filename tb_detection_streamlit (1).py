import streamlit as st
import zipfile
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ---------------------
# Helper Functions
# ---------------------
def unzip_dataset(uploaded_file):
    dataset_path = "dataset"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    return dataset_path

def get_class_distribution(dataset_path):
    class_counts = {}
    for root, dirs, files in os.walk(dataset_path):
        if len(files) > 0:
            label = os.path.basename(root)
            img_files = [f for f in files if os.path.isfile(os.path.join(root, f))]
            if len(img_files) > 0:
                class_counts[label] = class_counts.get(label, 0) + len(img_files)
    return class_counts

# ---------------------
# Streamlit App
# ---------------------
st.title("🫁 Tuberculosis Detection using Deep Learning")
menu = ["Introduction", "EDA", "Training", "Evaluation", "Prediction"]
choice = st.sidebar.selectbox("Navigation", menu)

if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = None
if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = None
if "train_gen" not in st.session_state:
    st.session_state.train_gen = None
if "val_gen" not in st.session_state:
    st.session_state.val_gen = None

# ---------------------
# Introduction
# ---------------------
if choice == "Introduction":
    st.subheader("Project Overview")
    st.write("""
    This app demonstrates Tuberculosis detection using Deep Learning.
    You can upload a dataset of chest X-ray images, explore it, train models (ResNet50, VGG16, EfficientNetB0),
    evaluate performance, and make predictions.
    """)
    uploaded_file = st.file_uploader("Upload a dataset (.zip)", type="zip")
    if uploaded_file:
        st.session_state.dataset_path = unzip_dataset(uploaded_file)
        st.success("Dataset uploaded and extracted successfully!")

# ---------------------
# EDA
# ---------------------
if choice == "EDA" and st.session_state.dataset_path:
    st.subheader("Exploratory Data Analysis")
    dataset_path = st.session_state.dataset_path

    class_counts = get_class_distribution(dataset_path)
    st.write("Class Distribution:", class_counts)

    fig, ax = plt.subplots()
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), ax=ax)
    ax.set_title("Class Distribution")
    st.pyplot(fig)

    st.write("Sample Images:")
    num_images = st.slider("Images per class", 4, 20, 8)

    for c in class_counts.keys():
        c_dir = None
        # find actual folder path
        for root, dirs, files in os.walk(dataset_path):
            if os.path.basename(root) == c:
                c_dir = root
                break
        if c_dir:
            files = [f for f in os.listdir(c_dir) if os.path.isfile(os.path.join(c_dir, f))]
            sample_files = files[:num_images]
            cols = st.columns(min(4, len(sample_files)))
            for idx, f in enumerate(sample_files):
                img_path = os.path.join(c_dir, f)
                try:
                    img = Image.open(img_path)
                    cols[idx % 4].image(img, caption=c, use_column_width=True)
                except:
                    continue

# ---------------------
# Training
# ---------------------
if choice == "Training" and st.session_state.dataset_path:
    st.subheader("Train Model")
    dataset_path = st.session_state.dataset_path

    model_name = st.selectbox("Select Model", ["ResNet50", "VGG16", "EfficientNetB0"])
    epochs = st.slider("Epochs", 1, 20, 5)
    val_split = st.slider("Validation Split %", 10, 50, 20) / 100.0
    img_size = (224, 224)

    # Detect train/val/test
    train_dir, val_dir, test_dir = None, None, None
    if os.path.isdir(os.path.join(dataset_path, "train")):
        train_dir = os.path.join(dataset_path, "train")
        if os.path.isdir(os.path.join(dataset_path, "val")):
            val_dir = os.path.join(dataset_path, "val")
        else:
            # Split train into train/val
            tmp_train, tmp_val = "split_train", "split_val"
            if os.path.exists(tmp_train): shutil.rmtree(tmp_train)
            if os.path.exists(tmp_val): shutil.rmtree(tmp_val)
            shutil.copytree(train_dir, tmp_train)
            shutil.copytree(train_dir, tmp_val)
            train_dir, val_dir = tmp_train, tmp_val
        if os.path.isdir(os.path.join(dataset_path, "test")):
            test_dir = os.path.join(dataset_path, "test")
    else:
        # Flat dataset, split manually
        tmp_base = "split_dataset"
        if os.path.exists(tmp_base): shutil.rmtree(tmp_base)
        os.makedirs(tmp_base, exist_ok=True)
        for c in os.listdir(dataset_path):
            c_path = os.path.join(dataset_path, c)
            if os.path.isdir(c_path):
                files = [f for f in os.listdir(c_path) if os.path.isfile(os.path.join(c_path, f))]
                train_files, val_files = train_test_split(files, test_size=val_split, random_state=42)
                os.makedirs(os.path.join(tmp_base, "train", c), exist_ok=True)
                os.makedirs(os.path.join(tmp_base, "val", c), exist_ok=True)
                for f in train_files:
                    shutil.copy(os.path.join(c_path, f), os.path.join(tmp_base, "train", c, f))
                for f in val_files:
                    shutil.copy(os.path.join(c_path, f), os.path.join(tmp_base, "val", c, f))
        train_dir, val_dir = os.path.join(tmp_base, "train"), os.path.join(tmp_base, "val")

    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=32, class_mode="binary")
    val_gen = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=32, class_mode="binary")

    base_model = None
    if model_name == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=img_size + (3,))
    elif model_name == "VGG16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=img_size + (3,))
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=img_size + (3,))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=preds)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    if st.button("Start Training"):
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
        st.session_state.model = model
        st.session_state.history = history
        st.session_state.train_gen = train_gen
        st.session_state.val_gen = val_gen
        st.success("Training completed!")

# ---------------------
# Evaluation
# ---------------------
if choice == "Evaluation" and st.session_state.model:
    st.subheader("Model Evaluation")
    model = st.session_state.model
    val_gen = st.session_state.val_gen
    preds = (model.predict(val_gen) > 0.5).astype(int)
    y_true = val_gen.classes

    report = classification_report(y_true, preds, target_names=list(val_gen.class_indices.keys()), output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=val_gen.class_indices.keys(), yticklabels=val_gen.class_indices.keys(), ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    fpr, tpr, _ = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# ---------------------
# Prediction
# ---------------------
if choice == "Prediction" and st.session_state.model:
    st.subheader("Make a Prediction")
    uploaded_img = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = st.session_state.model.predict(img_array)
        label = "Tuberculosis" if pred[0][0] > 0.5 else "Normal"
        st.write(f"### Prediction: {label}")
