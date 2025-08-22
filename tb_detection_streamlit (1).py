import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---------------- SAFE CLASSWISE SPLIT ---------------- #
def safe_classwise_split(dataset_path, val_split=0.2):
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    train_files, val_files = [], []

    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]

        if len(files) > 1:
            tr, vl = train_test_split(files, test_size=val_split, random_state=42)
            train_files.extend(tr)
            val_files.extend(vl)
        else:
            # if only one image, keep it in train
            train_files.extend(files)

    return train_files, val_files

# ---------------- STREAMLIT APP ---------------- #
st.title("🫁 Tuberculosis Detection from Chest X-rays")

# Sidebar options
section = st.sidebar.radio("Navigate", ["Upload Dataset", "EDA", "Train Model", "Evaluate Model", "Predict"])

# Cache paths
BASE_DIR = "dataset"
EXTRACTED_DIR = "data_extracted"

if section == "Upload Dataset":
    st.header("📂 Upload your dataset (.zip)")

    uploaded_file = st.file_uploader("Upload a ZIP file of dataset", type=["zip"])
    if uploaded_file is not None:
        if os.path.exists(EXTRACTED_DIR):
            shutil.rmtree(EXTRACTED_DIR)
        os.makedirs(EXTRACTED_DIR, exist_ok=True)

        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(EXTRACTED_DIR)

        st.success("✅ Dataset extracted successfully!")

        # detect folders
        subdirs = os.listdir(EXTRACTED_DIR)
        st.write("Detected folders:", subdirs)

if section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    if not os.path.exists(EXTRACTED_DIR):
        st.warning("⚠️ Please upload a dataset first.")
    else:
        # Count images per class
        class_counts = {}
        for root, dirs, files in os.walk(EXTRACTED_DIR):
            if len(files) > 0:
                label = os.path.basename(root)
                class_counts[label] = class_counts.get(label, 0) + len(files)

        df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
        st.bar_chart(df.set_index("Class"))

        # Preview images
        num_images = st.slider("Number of images per class to preview", 1, 5, 3)

        for cls in class_counts.keys():
            cls_path = None
            for root, dirs, files in os.walk(EXTRACTED_DIR):
                if os.path.basename(root) == cls:
                    cls_path = root
                    break
            if cls_path:
                imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path)[:num_images]]
                st.subheader(f"Class: {cls}")
                cols = st.columns(len(imgs))
                for i, img_path in enumerate(imgs):
                    img = Image.open(img_path)
                    cols[i].image(img, caption=cls, width=120)

if section == "Train Model":
    st.header("🧠 Train a Deep Learning Model")

    if not os.path.exists(EXTRACTED_DIR):
        st.warning("⚠️ Please upload a dataset first.")
    else:
        model_choice = st.selectbox("Choose model", ["ResNet50", "VGG16", "EfficientNetB0"])
        epochs = st.slider("Epochs", 1, 20, 5)
        val_split = st.slider("Validation Split (if no val/ folder)", 0.1, 0.5, 0.2)

        if st.button("Start Training"):
            # Detect structure
            train_dir = os.path.join(EXTRACTED_DIR, "train")
            val_dir = os.path.join(EXTRACTED_DIR, "val")

            if not os.path.exists(train_dir):
                # If flat dataset: move everything into train and split
                train_dir = os.path.join(EXTRACTED_DIR, "train_split")
                val_dir = os.path.join(EXTRACTED_DIR, "val_split")
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(val_dir, exist_ok=True)

                files_train, files_val = safe_classwise_split(EXTRACTED_DIR, val_split)

                for f in files_train:
                    cls = os.path.basename(os.path.dirname(f))
                    dest = os.path.join(train_dir, cls)
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy(f, dest)

                for f in files_val:
                    cls = os.path.basename(os.path.dirname(f))
                    dest = os.path.join(val_dir, cls)
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy(f, dest)

            # Generators
            datagen = ImageDataGenerator(rescale=1./255)

            train_gen = datagen.flow_from_directory(
                train_dir, target_size=(224,224), batch_size=32, class_mode="binary"
            )
            val_gen = datagen.flow_from_directory(
                val_dir, target_size=(224,224), batch_size=32, class_mode="binary"
            )

            # Model
            if model_choice == "ResNet50":
                base = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))
            elif model_choice == "VGG16":
                base = VGG16(weights=None, include_top=False, input_shape=(224,224,3))
            else:
                base = EfficientNetB0(weights=None, include_top=False, input_shape=(224,224,3))

            model = Sequential([
                base,
                GlobalAveragePooling2D(),
                Dropout(0.3),
                Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

            st.success("✅ Training complete!")
            st.line_chart(history.history["accuracy"])
            st.line_chart(history.history["val_accuracy"])

            model.save("tb_model.h5")
            st.info("💾 Model saved as tb_model.h5")

if section == "Evaluate Model":
    st.header("📈 Evaluate the Trained Model")
    st.info("Coming soon: load model, compute metrics on test set")

if section == "Predict":
    st.header("🔎 Predict from a Chest X-ray")

    from tensorflow.keras.models import load_model
    if os.path.exists("tb_model.h5"):
        model = load_model("tb_model.h5")

        uploaded_img = st.file_uploader("Upload an X-ray image", type=["jpg","png","jpeg"])
        if uploaded_img:
            img = Image.open(uploaded_img).convert("RGB").resize((224,224))
            arr = np.expand_dims(np.array(img)/255.0, axis=0)
            pred = model.predict(arr)[0][0]
            label = "Tuberculosis" if pred > 0.5 else "Normal"
            st.image(img, caption=f"Prediction: {label}", width=300)
    else:
        st.warning("⚠️ Please train the model first.")
