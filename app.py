import streamlit as st
import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="TB Detection System", layout="wide")

# ==============================
# Caching: Load MobileNetV2 once
# ==============================
@st.cache_resource
def get_base_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Load model from file if exists, else create
@st.cache_resource
def load_or_create_model():
    if os.path.exists("model.h5"):
        return load_model("model.h5")
    return get_base_model()

model = load_or_create_model()

# =========================
# Dataset splitting function
# =========================
def split_dataset(tb_dir, normal_dir, output_dir, train_ratio, val_ratio, test_ratio):
    for category, src_path in [('TB', tb_dir), ('NORMAL', normal_dir)]:
        images = os.listdir(src_path)
        random.shuffle(images)
        total = len(images)
        train_end = int(train_ratio * total)
        val_end = int((train_ratio + val_ratio) * total)

        sets = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for set_name, files in sets.items():
            dest_path = os.path.join(output_dir, set_name, category)
            os.makedirs(dest_path, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(src_path, f), os.path.join(dest_path, f))

# =======================
# Augmentation preview
# =======================
def preview_augmentation(data_dir):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    img_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_paths.append(os.path.join(root, file))
    sample_paths = random.sample(img_paths, min(3, len(img_paths)))
    st.write("### Augmentation Preview")
    cols = st.columns(len(sample_paths))
    for idx, img_path in enumerate(sample_paths):
        img = Image.open(img_path).resize((224, 224))
        x = np.expand_dims(np.array(img), 0)
        aug_iter = datagen.flow(x, batch_size=1)
        aug_img = next(aug_iter)[0].astype(np.uint8)
        cols[idx].image(aug_img, caption=f"Augmented {idx+1}")

# =======================
# Streamlit UI
# =======================
tabs = st.tabs(["📜 Introduction", "📂 Data Processing", "🧠 Model Training", "🔍 Prediction"])

# 1️⃣ Introduction Tab
with tabs[0]:
    st.title("Tuberculosis Detection Using Deep Learning")
    st.markdown("""
    **By the end of this project, you will achieve:**
    - ✅ A preprocessed and augmented dataset ready for deep learning  
    - ✅ A trained MobileNetV2 CNN model for TB detection  
    - ✅ A deployed TB detection system accessible via Streamlit  
    - ✅ A fully functional application hosted on AWS  
    """)

# 2️⃣ Data Processing Tab
with tabs[1]:
    st.header("Upload and Process Dataset")
    uploaded_zip = st.file_uploader("Upload ZIP containing TB & NORMAL folders", type="zip")
    train_ratio = st.slider("Train %", 50, 80, 70)
    val_ratio = st.slider("Validation %", 10, 30, 15)
    test_ratio = 100 - train_ratio - val_ratio

    if uploaded_zip and st.button("Process Dataset"):
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall("dataset")
        tb_dir = os.path.join("dataset", "TB")
        normal_dir = os.path.join("dataset", "NORMAL")
        if not (os.path.exists(tb_dir) and os.path.exists(normal_dir)):
            st.error("ZIP must contain 'TB' and 'NORMAL' folders.")
        else:
            if os.path.exists("processed"): shutil.rmtree("processed")
            split_dataset(tb_dir, normal_dir, "processed",
                          train_ratio/100, val_ratio/100, test_ratio/100)
            st.success("Dataset processed successfully!")
            preview_augmentation("processed/train/TB")

# 3️⃣ Model Training Tab
with tabs[2]:
    st.header("Train / Fine-tune Model")
    if st.button("Train Model"):
        if not os.path.exists("processed/train"):
            st.error("Please process dataset first.")
        else:
            datagen = ImageDataGenerator(rescale=1./255)
            train_gen = datagen.flow_from_directory("processed/train", target_size=(224, 224), batch_size=16, class_mode='binary')
            val_gen = datagen.flow_from_directory("processed/val", target_size=(224, 224), batch_size=16, class_mode='binary')

            with st.spinner("Training in progress..."):
                history = model.fit(train_gen, validation_data=val_gen, epochs=3)
                model.save("model.h5")
            st.success("Model trained and saved!")

# 4️⃣ Prediction Tab
with tabs[3]:
    st.header("Upload X-ray for TB Detection")
    uploaded_img = st.file_uploader("Upload chest X-ray", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, 0)
        pred = model.predict(img_array)[0][0]
        label = "TB" if pred > 0.5 else "Normal"
        st.image(img, caption=f"Prediction: {label} ({pred:.2f})", use_column_width=True)
