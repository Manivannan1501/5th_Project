import streamlit as st
import os
import zipfile
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Force headless OpenCV for Streamlit Cloud / server environments
try:
    import cv2
except ImportError:
    import pip
    pip.main(["install", "opencv-python-headless"])
    import cv2

from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tempfile


# -----------------------------
# Utility Functions
# -----------------------------
def extract_zip(uploaded_file, extract_to):
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def split_dataset(source_dir_tb, source_dir_normal, target_dir, train_ratio, val_ratio, test_ratio):
    classes = ["TB", "Normal"]
    split_ratio = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    random.seed(42)

    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

    for cls, src_path in zip(classes, [source_dir_tb, source_dir_normal]):
        images = os.listdir(src_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(split_ratio["train"] * total)
        val_end = train_end + int(split_ratio["val"] * total)

        data_splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, split_images in data_splits.items():
            for img in split_images:
                src = os.path.join(src_path, img)
                dst = os.path.join(target_dir, split, cls, img)
                if os.path.isfile(src):
                    shutil.copy(src, dst)

def preprocess_and_clean(data_dir):
    IMG_SIZE = (224, 224)
    splits = ["train", "val", "test"]
    classes = ["Normal", "TB"]

    for split in splits:
        for cls in classes:
            folder = os.path.join(data_dir, split, cls)
            if os.path.isdir(folder):
                for img_file in tqdm(os.listdir(folder), desc=f"{split}/{cls}"):
                    path = os.path.join(folder, img_file)
                    if os.path.isfile(path):
                        img = cv2.imread(path)
                        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                            os.remove(path)
                        else:
                            img = cv2.resize(img, IMG_SIZE)
                            cv2.imwrite(path, img)

def plot_class_distribution(data_dir, title):
    classes = []
    counts = []
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            classes.append(cls)
            counts.append(len(os.listdir(cls_path)))

    fig, ax = plt.subplots()
    sns.barplot(x=classes, y=counts, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    st.pyplot(fig)

def show_sample_images(data_dir, class_name, n=5):
    path = os.path.join(data_dir, class_name)
    images = os.listdir(path)[:n]
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i, img_name in enumerate(images):
        img = cv2.imread(os.path.join(path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(class_name)
        axes[i].axis('off')
    st.pyplot(fig)

def build_model(model_name, input_shape=(224, 224, 3)):
    if model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Invalid model name")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="TB Detection", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Data Preparation", "Training", "Evaluation", "Prediction"])

if page == "Introduction":
    st.title("Tuberculosis Detection Using Deep Learning")
    st.write("""
        This app allows you to train and evaluate deep learning models for detecting Tuberculosis from chest X-rays.
        You can upload a dataset, choose a CNN architecture, train the model, and test predictions.
    """)

elif page == "Data Preparation":
    st.header("Upload and Split Dataset")
    uploaded_file = st.file_uploader("Upload a ZIP file containing 'TB' and 'Normal' folders", type=["zip"])
    train_ratio = st.slider("Train %", 50, 90, 80)
    val_ratio = st.slider("Validation %", 5, 30, 10)
    test_ratio = 100 - train_ratio - val_ratio
    st.write(f"Test %: {test_ratio}")

    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        extract_zip(uploaded_file, temp_dir)
        tb_dir = os.path.join(temp_dir, "TB")
        normal_dir = os.path.join(temp_dir, "Normal")
        target_dir = os.path.join(temp_dir, "split_dataset")

        split_dataset(tb_dir, normal_dir, target_dir, train_ratio/100, val_ratio/100, test_ratio/100)
        preprocess_and_clean(target_dir)

        st.success("Dataset prepared successfully!")

        st.subheader("Train Set Distribution")
        plot_class_distribution(os.path.join(target_dir, "train"), "Train Set Distribution")

        st.subheader("Sample TB Images")
        show_sample_images(os.path.join(target_dir, "train"), "TB")

        st.subheader("Sample Normal Images")
        show_sample_images(os.path.join(target_dir, "train"), "Normal")

        st.session_state["data_dir"] = target_dir

elif page == "Training":
    if "data_dir" not in st.session_state:
        st.warning("Please prepare the dataset first in 'Data Preparation'.")
    else:
        st.header("Train Model")
        model_choice = st.selectbox("Choose model architecture", ["ResNet50", "VGG16", "EfficientNetB0"])
        epochs = st.number_input("Epochs", 1, 50, 5)
        batch_size = st.number_input("Batch Size", 8, 64, 16)

        if st.button("Start Training"):
            datagen = ImageDataGenerator(rescale=1./255)
            train_gen = datagen.flow_from_directory(os.path.join(st.session_state["data_dir"], "train"),
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
            val_gen = datagen.flow_from_directory(os.path.join(st.session_state["data_dir"], "val"),
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

            model = build_model(model_choice)
            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

            st.session_state["model"] = model
            st.session_state["history"] = history
            st.success("Training completed!")

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history['accuracy'], label='Train Accuracy')
            ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
            ax[0].legend()
            ax[0].set_title("Accuracy")

            ax[1].plot(history.history['loss'], label='Train Loss')
            ax[1].plot(history.history['val_loss'], label='Val Loss')
            ax[1].legend()
            ax[1].set_title("Loss")

            st.pyplot(fig)

elif page == "Evaluation":
    if "model" not in st.session_state:
        st.warning("Please train the model first.")
    else:
        st.header("Model Evaluation")
        datagen = ImageDataGenerator(rescale=1./255)
        test_gen = datagen.flow_from_directory(os.path.join(st.session_state["data_dir"], "test"),
                                               target_size=(224, 224),
                                               batch_size=1,
                                               class_mode='categorical',
                                               shuffle=False)

        preds = st.session_state["model"].predict(test_gen)
        y_pred = np.argmax(preds, axis=1)
        y_true = test_gen.classes

        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

elif page == "Prediction":
    if "model" not in st.session_state:
        st.warning("Please train the model first.")
    else:
        st.header("Predict a Chest X-ray")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img, (224, 224)) / 255.0
            img_array = np.expand_dims(img_resized, axis=0)

            preds = st.session_state["model"].predict(img_array)
            class_names = list(st.session_state["model"].classes_.keys()) if hasattr(st.session_state["model"], "classes_") else ["TB", "Normal"]
            predicted_class = class_names[np.argmax(preds)]
            confidence = np.max(preds) * 100

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Prediction: {predicted_class} ({confidence:.2f}%)", use_column_width=True)
