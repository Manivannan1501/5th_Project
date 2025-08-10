import streamlit as st
import os
import zipfile
import tempfile
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # Use opencv-python-headless in requirements.txt
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# ---------------------------
# Helper: find TB/NORMAL dirs anywhere
# ---------------------------
def find_class_dir(base_path, name):
    for root, dirs, _ in os.walk(base_path):
        if name in dirs:
            return os.path.join(root, name)
    return None


# ---------------------------
# Helper: split dataset
# ---------------------------
def split_dataset(tb_dir, normal_dir, target_dir, train_ratio, val_ratio, test_ratio):
    os.makedirs(target_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_dir, split, "TB"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, "NORMAL"), exist_ok=True)

    def copy_files(src_path, dest_path, label, split_ratio):
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
            shutil.copy(
                os.path.join(src_path, img),
                os.path.join(dest_path, subset, label, img)
            )

    copy_files(tb_dir, target_dir, "TB", (train_ratio, val_ratio, test_ratio))
    copy_files(normal_dir, target_dir, "NORMAL", (train_ratio, val_ratio, test_ratio))


# ---------------------------
# Helper: build model
# ---------------------------
def build_model(arch, input_shape=(224, 224, 3)):
    if arch == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif arch == "VGG16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------
# Cache: extract dataset once per upload
# ---------------------------
@st.cache_resource
def extract_dataset(uploaded_file):
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "dataset.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    tb_dir = find_class_dir(tmp_dir, "TB")
    normal_dir = find_class_dir(tmp_dir, "NORMAL")

    if not tb_dir or not normal_dir:
        return None, None, None

    return tb_dir, normal_dir, tmp_dir


# ---------------------------
# Cache: split dataset based on ratios
# ---------------------------
@st.cache_resource
def prepare_split(tb_dir, normal_dir, train_ratio, val_ratio, test_ratio):
    target_dir = tempfile.mkdtemp()
    split_dataset(tb_dir, normal_dir, target_dir,
                  train_ratio/100, val_ratio/100, test_ratio/100)
    return target_dir


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Tuberculosis Detection App")
st.write("Upload a dataset ZIP containing `TB` and `NORMAL` folders (anywhere inside).")

uploaded_file = st.file_uploader("Upload dataset (ZIP)", type=["zip"])
train_ratio = st.slider("Train ratio (%)", 50, 90, 70)
val_ratio = st.slider("Validation ratio (%)", 5, 30, 15)
test_ratio = 100 - train_ratio - val_ratio
model_choice = st.selectbox("Model Architecture", ["ResNet50", "VGG16", "EfficientNetB0"])
epochs = st.slider("Training Epochs", 1, 10, 3)

if uploaded_file is not None:
    tb_dir, normal_dir, base_tmp = extract_dataset(uploaded_file)

    if tb_dir is None:
        st.error("Could not find TB and NORMAL folders in ZIP.")
    else:
        target_dir = prepare_split(tb_dir, normal_dir, train_ratio, val_ratio, test_ratio)

        # Show counts
        for split in ["train", "val", "test"]:
            tb_count = len(os.listdir(os.path.join(target_dir, split, "TB")))
            normal_count = len(os.listdir(os.path.join(target_dir, split, "NORMAL")))
            st.write(f"**{split.capitalize()}** — TB: {tb_count}, NORMAL: {normal_count}")

        # Data generators
        datagen = ImageDataGenerator(rescale=1./255)
        train_gen = datagen.flow_from_directory(os.path.join(target_dir, "train"), target_size=(224, 224), batch_size=32, class_mode="binary")
        val_gen = datagen.flow_from_directory(os.path.join(target_dir, "val"), target_size=(224, 224), batch_size=32, class_mode="binary")
        test_gen = datagen.flow_from_directory(os.path.join(target_dir, "test"), target_size=(224, 224), batch_size=32, class_mode="binary", shuffle=False)

        # Build & train
        model = build_model(model_choice)
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

        # Plot training history
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history["accuracy"], label="train_acc")
        ax[0].plot(history.history["val_accuracy"], label="val_acc")
        ax[0].set_title("Accuracy")
        ax[0].legend()

        ax[1].plot(history.history["loss"], label="train_loss")
        ax[1].plot(history.history["val_loss"], label="val_loss")
        ax[1].set_title("Loss")
        ax[1].legend()
        st.pyplot(fig)

        # Evaluate
        preds = model.predict(test_gen)
        y_true = test_gen.classes
        y_pred = (preds > 0.5).astype(int).ravel()

        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "TB"], yticklabels=["NORMAL", "TB"])
        st.pyplot(fig)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, preds)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
