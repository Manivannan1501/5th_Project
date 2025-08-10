import streamlit as st
import os
import zipfile
import tempfile
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # use opencv-python-headless in requirements.txt
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
            shutil.copy(os.path.join(src_path, img),
                        os.path.join(dest_path, subset, os.path.basename(src_path), img))

    copy_files(tb_dir, target_dir, (train_ratio, val_ratio, test_ratio))
    copy_files(normal_dir, target_dir, (train_ratio, val_ratio, test_ratio))


# ---------------------------
# Model creation helper
# ---------------------------
def build_model(model_name, input_shape=(224, 224, 3)):
    if model_name == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "VGG16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Invalid model name.")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Tuberculosis Detection using Deep Learning")
st.write("Upload a dataset ZIP containing `TB` and `NORMAL` folders at root.")

uploaded_file = st.file_uploader("Upload dataset (ZIP)", type=["zip"])
train_ratio = st.slider("Train ratio (%)", 50, 90, 70)
val_ratio = st.slider("Validation ratio (%)", 5, 30, 15)
test_ratio = 100 - train_ratio - val_ratio
model_choice = st.selectbox("Choose Model", ["ResNet50", "VGG16", "EfficientNetB0"])
epochs = st.slider("Epochs", 1, 20, 5)

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save uploaded file
        zip_path = os.path.join(tmp_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Paths
        tb_dir = os.path.join(tmp_dir, "TB")
        normal_dir = os.path.join(tmp_dir, "NORMAL")
        target_dir = os.path.join(tmp_dir, "split_data")

        if not os.path.exists(tb_dir) or not os.path.exists(normal_dir):
            st.error("ZIP must contain 'TB' and 'NORMAL' folders at root.")
        else:
            # Split dataset
            split_dataset(tb_dir, normal_dir, target_dir,
                          train_ratio/100, val_ratio/100, test_ratio/100)
            st.success("Dataset split completed!")

            # Data generators
            datagen = ImageDataGenerator(rescale=1.0/255)
            train_gen = datagen.flow_from_directory(os.path.join(target_dir, "train"),
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode="categorical")
            val_gen = datagen.flow_from_directory(os.path.join(target_dir, "val"),
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode="categorical")
            test_gen = datagen.flow_from_directory(os.path.join(target_dir, "test"),
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode="categorical",
                                                   shuffle=False)

            # Build and train model
            st.write(f"Training **{model_choice}**...")
            model = build_model(model_choice)
            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

            # Plot accuracy/loss
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history["accuracy"], label="Train Acc")
            ax[0].plot(history.history["val_accuracy"], label="Val Acc")
            ax[0].set_title("Accuracy")
            ax[0].legend()

            ax[1].plot(history.history["loss"], label="Train Loss")
            ax[1].plot(history.history["val_loss"], label="Val Loss")
            ax[1].set_title("Loss")
            ax[1].legend()
            st.pyplot(fig)

            # Evaluation
            preds = model.predict(test_gen)
            y_true = test_gen.classes
            y_pred = np.argmax(preds, axis=1)

            st.subheader("Classification Report")
            report = classification_report(y_true, y_pred, target_names=["NORMAL", "TB"], output_dict=True)
            st.json(report)

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "TB"], yticklabels=["NORMAL", "TB"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, preds[:, 1])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
