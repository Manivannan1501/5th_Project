"""
Final Streamlit app: upload ZIP -> prepare (cached) -> split (cached per-ratio) -> augment/train -> evaluate -> save -> predict
Notes:
- ZIP can contain TB and NORMAL folders anywhere inside (will be found automatically).
- Use opencv-python-headless in requirements.txt for cloud.
"""

import streamlit as st
import os
import zipfile
import tempfile
import random
import shutil
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # headless in requirements
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="TB Detection", layout="wide")

# -----------------------
# Intro / Learning outcomes
# -----------------------
st.title("🩺 Tuberculosis Detection — End-to-end")
st.header("Learning Outcomes")
st.markdown("""
By the end of this project, learners will achieve:

- **A preprocessed and augmented dataset ready for deep learning.**  
  Use the app to upload raw X-rays, clean, resize, split, and (optionally) augment images for model training.

- **Multiple CNN-based models trained and evaluated.**  
  Choose between ResNet50, VGG16, and EfficientNetB0. Train, view training curves, classification report, confusion matrix, and ROC-AUC.

- **A deployed TB detection system accessible via a Streamlit interface.**

- **A fully functional application hosted on AWS.**
""")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Train Model", "Predict / Inference", "About / README"])

# ---------------------------
# Helpers
# ---------------------------
def find_class_dir(base_path, name):
    for root, dirs, _ in os.walk(base_path):
        if name in dirs:
            return os.path.join(root, name)
    return None

def split_dataset(tb_dir, normal_dir, target_dir, train_ratio, val_ratio, test_ratio):
    os.makedirs(target_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_dir, split, "TB"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, "NORMAL"), exist_ok=True)

    def copy_files(src_path, dest_path, label, split_ratio):
        items = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        random.shuffle(items)
        n_total = len(items)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        for i, fname in enumerate(items):
            if i < n_train:
                subset = "train"
            elif i < n_train + n_val:
                subset = "val"
            else:
                subset = "test"
            shutil.copy(os.path.join(src_path, fname),
                        os.path.join(dest_path, subset, label, fname))

def build_model(arch, input_shape=(224,224,3)):
    if arch == "ResNet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif arch == "VGG16":
        base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

    x = base.output
    x = GlobalAveragePooling2D()(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=out)

    for layer in base.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------
# Caching helpers
# ---------------------------
@st.cache_resource(show_spinner=False)
def extract_dataset_from_zip_bytes(zip_bytes):
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "dataset.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)
    tb_dir = find_class_dir(tmp_dir, "TB") or find_class_dir(tmp_dir, "tb") or find_class_dir(tmp_dir, "Tb")
    normal_dir = find_class_dir(tmp_dir, "NORMAL") or find_class_dir(tmp_dir, "normal") or find_class_dir(tmp_dir, "Normal")
    return tb_dir, normal_dir, tmp_dir

# split cache keyed on (tb_dir_path, normal_dir_path, train_ratio, val_ratio, test_ratio)
@st.cache_resource(show_spinner=False)
def prepare_split_cached(tb_dir, normal_dir, train_ratio, val_ratio, test_ratio):
    # create a fresh temp dir for split outputs
    out_dir = tempfile.mkdtemp()
    split_dataset(tb_dir, normal_dir, out_dir, train_ratio, val_ratio, test_ratio)
    return out_dir

# ---------------------------
# TRAIN PAGE
# ---------------------------
if page == "Train Model":
    st.header("1. Upload dataset (ZIP)")
    st.markdown("ZIP must contain `TB` and `NORMAL` folders somewhere inside (root or nested).")
    uploaded = st.file_uploader("Upload dataset ZIP", type=["zip"])
    st.markdown("---")

    st.header("2. Split & Augmentation options")
    col1, col2 = st.columns(2)
    with col1:
        train_pct = st.slider("Train (%)", 50, 90, 70)
        val_pct = st.slider("Val (%)", 5, 30, 15)
        test_pct = 100 - train_pct - val_pct
        st.write(f"Test (%): {test_pct}")
    with col2:
        augment = st.checkbox("Enable Data Augmentation", value=True)
        batch_size = st.number_input("Batch size", 8, 64, 32)
        epochs = st.slider("Epochs", 1, 20, 5)
        model_arch = st.selectbox("Model architecture", ["ResNet50","VGG16","EfficientNetB0"])

    if uploaded is not None:
        # extract once (cached)
        zip_bytes = uploaded.getvalue()
        tb_dir, normal_dir, extracted_base = extract_dataset_from_zip_bytes(zip_bytes)

        if not tb_dir or not normal_dir:
            st.error("Could not find TB and NORMAL folders in the uploaded ZIP. Make sure folders exist (case-insensitive).")
        else:
            st.success("Found TB and NORMAL folders.")
            st.info(f"TB dir: {tb_dir}\nNORMAL dir: {normal_dir}")

            # prepare split (cached per-ratio)
            split_dir = prepare_split_cached(tb_dir, normal_dir, train_pct, val_pct, test_pct)
            st.write("Dataset split prepared (cached).")

            # show counts
            st.subheader("Class counts by split")
            counts = {}
            for s in ["train","val","test"]:
                t = len(os.listdir(os.path.join(split_dir, s, "TB")))
                n = len(os.listdir(os.path.join(split_dir, s, "NORMAL")))
                counts[s] = {"TB": t, "NORMAL": n}
                st.write(f"**{s}** — TB: {t} | NORMAL: {n}")

            # generator
            if augment:
                datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=15,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.05,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )
            else:
                datagen = ImageDataGenerator(rescale=1./255)

            train_gen = datagen.flow_from_directory(os.path.join(split_dir, "train"), target_size=(224,224), batch_size=batch_size, class_mode="binary")
            val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(os.path.join(split_dir, "val"), target_size=(224,224), batch_size=batch_size, class_mode="binary")
            test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(os.path.join(split_dir, "test"), target_size=(224,224), batch_size=1, class_mode="binary", shuffle=False)

            st.markdown("---")
            st.header("3. Train model")
            if st.button("Start Training"):
                with st.spinner("Training... (this may take time)"):
                    model = build_model(model_arch)
                    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

                # Show training plots
                fig, ax = plt.subplots(1,2,figsize=(12,4))
                ax[0].plot(history.history.get("accuracy", []), label="train_acc")
                ax[0].plot(history.history.get("val_accuracy", []), label="val_acc")
                ax[0].set_title("Accuracy"); ax[0].legend()
                ax[1].plot(history.history.get("loss", []), label="train_loss")
                ax[1].plot(history.history.get("val_loss", []), label="val_loss")
                ax[1].set_title("Loss"); ax[1].legend()
                st.pyplot(fig)

                # Evaluate on test set
                preds = model.predict(test_gen)
                y_true = test_gen.classes
                y_pred = (preds > 0.5).astype(int).ravel()

                st.subheader("Classification report")
                cr = classification_report(y_true, y_pred, target_names=["NORMAL","TB"], output_dict=True)
                st.json(cr)

                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL","TB"], yticklabels=["NORMAL","TB"], ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                st.pyplot(fig)

                # ROC
                try:
                    fpr, tpr, _ = roc_curve(y_true, preds.ravel())
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                    ax.plot([0,1],[0,1],"k--")
                    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
                    st.pyplot(fig)
                except Exception:
                    st.warning("Could not compute ROC curve (maybe single-class or insufficient data).")

                # Save model to temporary path and allow download
                model_save_path = os.path.join(tempfile.mkdtemp(), "tb_model.h5")
                model.save(model_save_path)
                st.success("Model trained and saved to temporary file.")
                with open(model_save_path, "rb") as f:
                    bytes_data = f.read()
                    st.download_button("Download trained model (.h5)", data=bytes_data, file_name="tb_model.h5")
                # store path in session for Prediction page (we'll keep bytes in session)
                st.session_state["trained_model_bytes"] = bytes_data
                st.session_state["class_mode"] = "binary"

# ---------------------------
# PREDICTION PAGE
# ---------------------------
elif page == "Predict / Inference":
    st.header("Use a trained model to predict a new X-ray image")
    if "trained_model_bytes" not in st.session_state:
        st.warning("No trained model found in session. Train a model first on the 'Train Model' page or upload a .h5 model below.")
        uploaded_model = st.file_uploader("Upload existing trained Keras model (.h5)", type=["h5"])
        if uploaded_model is not None:
            model_bytes = uploaded_model.getvalue()
            with open(os.path.join(tempfile.mkdtemp(), "user_model.h5"), "wb") as f:
                f.write(model_bytes)
            st.session_state["trained_model_bytes"] = model_bytes
            st.success("Model uploaded and cached for session.")
    else:
        # load model from bytes
        tmp_model_path = os.path.join(tempfile.mkdtemp(), "session_model.h5")
        with open(tmp_model_path, "wb") as f:
            f.write(st.session_state["trained_model_bytes"])
        model = load_model(tmp_model_path)

        uploaded_img = st.file_uploader("Upload chest X-ray image (jpg/png)", type=["jpg","jpeg","png"])
        if uploaded_img:
            # display
            st.image(uploaded_img, caption="Uploaded", use_column_width=True)
            # preprocess
            img = load_img(uploaded_img, target_size=(224,224))
            arr = img_to_array(img)/255.0
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr)[0][0]
            label = "TB" if pred > 0.5 else "NORMAL"
            confidence = float(pred) if pred > 0.5 else 1.0 - float(pred)
            st.metric("Prediction", f"{label} ({confidence*100:.2f}%)")

# ---------------------------
# About / README page
# ---------------------------
elif page == "About / README":
    st.header("Project README & Deployment")
    st.markdown("""
**Quick local run**
1. Create virtualenv, install requirements.txt
2. `streamlit run app.py`

**Data format**  
Upload a ZIP containing `TB` and `NORMAL` image folders anywhere inside (root or nested).

**Deployment**  
Use the provided `deploy.sh` for a quick EC2 setup or containerize the app with Docker for production.

**Notes**  
- For heavy training, use GPU-enabled instance or train offline and deploy only inference on AWS.
- Use `opencv-python-headless` for cloud environments.
""")
