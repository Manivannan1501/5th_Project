import streamlit as st
import os
import zipfile
import tempfile
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2  # from opencv-python-headless
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ---------------------------
# Helper: split dataset
# ---------------------------
def split_dataset(tb_dir, normal_dir, target_dir, train_ratio, val_ratio, test_ratio):
    os.makedirs(target_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_dir, split, "TB"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, "NORMAL"), exist_ok=True)

    def copy_files(src_path, dest_path, split_ratio):
        images = os.listdir(src_path)
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        for i, img in enumerate(images):
            if i < n_train:
                subset = "train"
            elif i < n_train + n_val:
                subset = "val"
            else:
                subset = "test"
            shutil.copy(os.path.join(src_path, img), os.path.join(dest_path, subset, os.path.basename(src_path), img))

    copy_files(tb_dir, target_dir, (train_ratio, val_ratio, test_ratio))
    copy_files(normal_dir, target_dir, (train_ratio, val_ratio, test_ratio))


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Tuberculosis Detection App")
st.write("Upload a dataset ZIP with `TB` and `NORMAL` folders.")

# Dataset upload
uploaded_file = st.file_uploader("Upload dataset (ZIP)", type=["zip"])
train_ratio = st.slider("Train ratio (%)", 50, 90, 70)
val_ratio = st.slider("Validation ratio (%)", 5, 30, 15)
test_ratio = 100 - train_ratio - val_ratio

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save uploaded file
        zip_path = os.path.join(tmp_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract zip
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Paths
        tb_dir = os.path.join(tmp_dir, "TB")
        normal_dir = os.path.join(tmp_dir, "NORMAL")
        target_dir = os.path.join(tmp_dir, "split_data")

        # Check folders
        if not os.path.exists(tb_dir) or not os.path.exists(normal_dir):
            st.error("ZIP must contain 'TB' and 'NORMAL' folders at root.")
        else:
            # Split dataset
            split_dataset(tb_dir, normal_dir, target_dir,
                          train_ratio/100, val_ratio/100, test_ratio/100)
            st.success("Dataset split completed!")

            # Optional: show counts
            for split in ["train", "val", "test"]:
                tb_count = len(os.listdir(os.path.join(target_dir, split, "TB")))
                normal_count = len(os.listdir(os.path.join(target_dir, split, "NORMAL")))
                st.write(f"**{split.capitalize()}** — TB: {tb_count}, NORMAL: {normal_count}")

            # Here you can call your model training / evaluation functions
