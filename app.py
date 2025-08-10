import streamlit as st
import os
import zipfile
import tempfile
import shutil
import random
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # ensure opencv-python-headless in requirements
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="TB Detection (MobileNetV2)", layout="wide")
st.title("🩺 TB Detection — MobileNetV2 (fast deployment)")
st.markdown("Upload dataset ZIP (contains `TB` and `NORMAL` folders anywhere). Use the tabs to prepare data, train (optional), and predict.")

# -------------------------
# Helpers (lightweight, safe)
# -------------------------
def find_class_dir(base_path, name):
    """Find directory named `name` (case-insensitive) anywhere under base_path."""
    name_lower = name.lower()
    for root, dirs, _ in os.walk(base_path):
        for d in dirs:
            if d.lower() == name_lower:
                return os.path.join(root, d)
    return None

def copy_split(src_dir, dst_root, label, train_ratio, val_ratio, test_ratio):
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    for i, fname in enumerate(files):
        if i < n_train:
            subset = "train"
        elif i < n_train + n_val:
            subset = "val"
        else:
            subset = "test"
        dst_dir = os.path.join(dst_root, subset, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

def split_dataset(tb_dir, normal_dir, out_root, train_ratio, val_ratio, test_ratio):
    # create structure and copy
    for s in ("train","val","test"):
        for lbl in ("TB","NORMAL"):
            os.makedirs(os.path.join(out_root, s, lbl), exist_ok=True)
    copy_split(tb_dir, out_root, "TB", train_ratio, val_ratio, test_ratio)
    copy_split(normal_dir, out_root, "NORMAL", train_ratio, val_ratio, test_ratio)

def build_mobilenetv2(input_shape=(224,224,3), lr=1e-4):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=out)
    # freeze base by default
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -------------------------
# Caching helpers
# -------------------------
@st.cache_data(show_spinner=False)
def extract_zip_cached(zip_bytes: bytes):
    tmp_dir = tempfile.mkdtemp()
    zpath = os.path.join(tmp_dir, "dataset.zip")
    with open(zpath, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(tmp_dir)
    tb = find_class_dir(tmp_dir, "TB")
    normal = find_class_dir(tmp_dir, "NORMAL")
    # also try lowercase/variations
    if not tb:
        tb = find_class_dir(tmp_dir, "tb")
    if not normal:
        normal = find_class_dir(tmp_dir, "normal")
    return tmp_dir, tb, normal

@st.cache_data(show_spinner=False)
def prepare_split_cached(base_tmp_dir: str, tb_dir: str, normal_dir: str, train_pct: int, val_pct: int):
    # new out dir for each ratio set (cache key includes params)
    out_dir = tempfile.mkdtemp()
    train_ratio = train_pct / 100.0
    val_ratio = val_pct / 100.0
    test_ratio = 1.0 - train_ratio - val_ratio
    split_dataset(tb_dir, normal_dir, out_dir, train_ratio, val_ratio, test_ratio)
    return out_dir

# -------------------------
# UI: Tabs
# -------------------------
tabs = st.tabs(["Introduction", "Data Processing", "Model & Train", "Predict / Inference"])

# 1) Introduction
with tabs[0]:
    st.header("Introduction & Learning Outcomes")
    st.markdown("""
    **Learning outcomes**
    - Preprocessed & optionally augmented dataset ready for DL.
    - MobileNetV2-based model for fast inference (optionally fine-tunable).
    - Streamlit interface for training and inference.
    - AWS-ready lightweight model for deployment.
    """)

# 2) Data Processing
with tabs[1]:
    st.header("Data Processing")
    st.markdown("Upload a ZIP file that contains `TB` and `NORMAL` folders anywhere inside (root or nested).")
    uploaded = st.file_uploader("Upload dataset ZIP", type=["zip"], key="data_zip")
    col1, col2 = st.columns(2)
    with col1:
        train_pct = st.slider("Train %", 50, 90, 70)
        val_pct = st.slider("Validation %", 5, 30, 15)
        test_pct = 100 - train_pct - val_pct
        st.info(f"Test % will be: {test_pct}%")
    with col2:
        preview_aug = st.checkbox("Preview augmentations", value=True)
        sample_count = st.number_input("Augment preview samples", min_value=3, max_value=12, value=6)
    if uploaded:
        zip_bytes = uploaded.getvalue()
        base_tmp, tb_dir, normal_dir = extract_zip_cached(zip_bytes)
        if not tb_dir or not normal_dir:
            st.error("Could not find TB and NORMAL folders inside the ZIP. Ensure folders exist (case-insensitive).")
        else:
            st.success("Found TB & NORMAL folders.")
            st.write("Paths found (for info):")
            st.write("TB:", tb_dir)
            st.write("NORMAL:", normal_dir)
            if st.button("Prepare / Split Dataset (run once)"):
                with st.spinner("Splitting dataset..."):
                    split_dir = prepare_split_cached(base_tmp, tb_dir, normal_dir, train_pct, val_pct)
                st.success("Dataset split prepared (cached).")
                # show counts
                for s in ("train","val","test"):
                    tb_count = len(os.listdir(os.path.join(split_dir, s, "TB")))
                    n_count = len(os.listdir(os.path.join(split_dir, s, "NORMAL")))
                    st.write(f"**{s}** — TB: {tb_count} | NORMAL: {n_count}")
                st.session_state["split_dir"] = split_dir

            # augmentation preview (sample random images + show augmented variants)
            if preview_aug:
                st.subheader("Augmentation preview (random samples)")
                # pick some sample files from TB or NORMAL depending on availability
                sample_pool = []
                if tb_dir:
                    sample_pool += [os.path.join(tb_dir, f) for f in os.listdir(tb_dir)]
                if normal_dir:
                    sample_pool += [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]
                sample_pool = [p for p in sample_pool if os.path.isfile(p)]
                if len(sample_pool) == 0:
                    st.warning("No images found to preview.")
                else:
                    sample_pool = random.sample(sample_pool, min(len(sample_pool), sample_count))
                    # augmentation generator
                    aug_gen = ImageDataGenerator(
                        rotation_range=15,
                        width_shift_range=0.08,
                        height_shift_range=0.08,
                        shear_range=0.05,
                        zoom_range=0.08,
                        horizontal_flip=True,
                        fill_mode='nearest'
                    )
                    cols = st.columns(len(sample_pool))
                    for i, path in enumerate(sample_pool):
                        with cols[i]:
                            try:
                                img = load_img(path, target_size=(224,224))
                                arr = img_to_array(img)
                                arr = np.expand_dims(arr, 0)
                                st.image(arr[0].astype('uint8'), caption=os.path.basename(path))
                                # show a few augmented variants
                                aug_imgs = aug_gen.flow(arr, batch_size=1)
                                aug_fig, ax = plt.subplots(1,3, figsize=(6,2))
                                for j in range(3):
                                    a = next(aug_imgs)[0].astype('uint8')
                                    ax[j].imshow(a)
                                    ax[j].axis('off')
                                st.pyplot(aug_fig)
                            except Exception as e:
                                st.write("Error previewing", e)

# 3) Model & Train
with tabs[2]:
    st.header("Model & Training")
    st.markdown("MobileNetV2 base (ImageNet) is provided for immediate inference. Optionally retrain/fine-tune on your dataset.")
    colA, colB = st.columns(2)
    with colA:
        use_prebuilt = st.checkbox("Use built-in MobileNetV2 (ImageNet base)", value=True)
        upload_model_file = st.file_uploader("Or upload a trained Keras model (.h5)", type=["h5"], key="upload_model")
    with colB:
        fine_tune = st.checkbox("Enable fine-tuning (unfreeze base conv layers) during retrain", value=False)
        retrain_epochs = st.number_input("Retrain epochs", min_value=1, max_value=50, value=3)
        retrain_batch = st.number_input("Retrain batch size", min_value=4, max_value=64, value=16)

    # Prepare a model object (not heavy until training)
    model_for_inference = None
    if upload_model_file is not None:
        # load user-supplied model bytes to a temp file
        saved = tempfile.mkdtemp()
        fpath = os.path.join(saved, "uploaded_model.h5")
        with open(fpath, "wb") as f:
            f.write(upload_model_file.getvalue())
        try:
            model_for_inference = load_model(fpath)
            st.success("Uploaded model loaded for inference.")
            st.session_state["model_bytes"] = upload_model_file.getvalue()
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
    elif use_prebuilt:
        # create a ready-to-use MobileNetV2 model (ImageNet base, frozen)
        model_for_inference = build_mobilenetv2()
        st.info("Built-in MobileNetV2 ready for inference (ImageNet base; not TB-trained).")

    # Show model summary (collapsed)
    if model_for_inference is not None:
        if st.checkbox("Show model summary (text)", value=False):
            with st.expander("Model summary"):
                model_for_inference.summary(print_fn=lambda s: st.text(s))

    # Retrain block (runs only when button clicked)
    if "split_dir" not in st.session_state:
        st.info("No dataset split prepared yet. Go to Data Processing tab and click 'Prepare / Split Dataset'.")
    else:
        split_dir = st.session_state["split_dir"]
        st.write("Split dataset found:", split_dir)
        if st.button("Retrain / Fine-tune model on dataset"):
            if model_for_inference is None:
                st.error("No model available for retraining. Use built-in model or upload a .h5 model.")
            else:
                with st.spinner("Preparing data generators..."):
                    # training generators (binary)
                    if st.session_state.get("preview_aug_enabled", True):
                        train_datagen = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=15,
                            width_shift_range=0.08,
                            height_shift_range=0.08,
                            shear_range=0.05,
                            zoom_range=0.08,
                            horizontal_flip=True,
                            fill_mode='nearest'
                        )
                    else:
                        train_datagen = ImageDataGenerator(rescale=1./255)
                    train_gen = train_datagen.flow_from_directory(os.path.join(split_dir, "train"),
                                                                  target_size=(224,224), batch_size=retrain_batch, class_mode="binary")
                    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(os.path.join(split_dir, "val"),
                                                                                     target_size=(224,224), batch_size=retrain_batch, class_mode="binary")
                # optionally unfreeze the base
                if fine_tune:
                    for layer in model_for_inference.layers:
                        layer.trainable = True
                    st.info("Base model unfrozen for fine-tuning.")
                else:
                    for layer in model_for_inference.layers:
                        layer.trainable = True if isinstance(layer, Dense) else layer.trainable  # keep dense trainable
                # recompile with small lr
                model_for_inference.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
                with st.spinner("Training (this may take time)..."):
                    history = model_for_inference.fit(train_gen, validation_data=val_gen, epochs=retrain_epochs)
                st.success("Retraining complete.")
                # evaluate
                test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(os.path.join(split_dir, "test"),
                                                                                  target_size=(224,224), batch_size=1, class_mode="binary", shuffle=False)
                preds = model_for_inference.predict(test_gen)
                y_true = test_gen.classes
                y_pred = (preds > 0.5).astype(int).ravel()
                st.subheader("Classification report")
                st.text(classification_report(y_true, y_pred))
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", xticklabels=["NORMAL","TB"], yticklabels=["NORMAL","TB"], ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                st.pyplot(fig)
                # save model bytes to session for Predict tab
                save_dir = tempfile.mkdtemp()
                model_path = os.path.join(save_dir, "tb_mobilenetv2.h5")
                model_for_inference.save(model_path)
                with open(model_path, "rb") as f:
                    mb = f.read()
                st.session_state["model_bytes"] = mb
                st.success("Trained/fine-tuned model cached in session (you can download it from the Predict tab).")

# 4) Predict / Inference
with tabs[3]:
    st.header("Prediction / Inference")
    st.markdown("Use a trained model (uploaded or retrained) or built-in MobileNetV2 for inference.")
    model_bytes = st.session_state.get("model_bytes", None)
    model_for_pred = None
    if model_bytes:
        # write bytes to temp file and load
        tmpd = tempfile.mkdtemp()
        p = os.path.join(tmpd, "session_model.h5")
        with open(p, "wb") as f:
            f.write(model_bytes)
        try:
            model_for_pred = load_model(p)
            st.success("Loaded model from session.")
            st.download_button("Download cached model (.h5)", data=model_bytes, file_name="tb_model.h5")
        except Exception as e:
            st.error(f"Could not load cached model: {e}")
    else:
        if st.checkbox("Use built-in MobileNetV2 (ImageNet base) for quick demo"):
            model_for_pred = build_mobilenetv2()
            st.info("Using ImageNet-only MobileNetV2 (not TB-trained) — demo only.")

    # user can also upload a model directly here
    uploaded_model = st.file_uploader("Or upload a .h5 model for prediction", type=["h5"], key="model_for_pred_upload")
    if uploaded_model is not None:
        tmpd2 = tempfile.mkdtemp()
        p2 = os.path.join(tmpd2, "uploaded_model.h5")
        with open(p2, "wb") as f:
            f.write(uploaded_model.getvalue())
        try:
            model_for_pred = load_model(p2)
            st.success("Uploaded model loaded for prediction.")
        except Exception as e:
            st.error(f"Failed loading uploaded model: {e}")

    if model_for_pred is None:
        st.info("No model available for prediction. Retrain or upload a model.")
    else:
        st.subheader("Predict a single X-ray")
        img_file = st.file_uploader("Upload chest X-ray image (jpg/png)", type=["jpg","jpeg","png"], key="pred_img")
        if img_file is not None:
            # display
            st.image(img_file, caption="Input image", use_column_width=True)
            # preprocess & predict
            try:
                img = load_img(img_file, target_size=(224,224))
                arr = img_to_array(img) / 255.0
                arr = np.expand_dims(arr, 0)
                pred = model_for_pred.predict(arr)[0][0]
                label = "TB" if pred > 0.5 else "NORMAL"
                confidence = pred if pred > 0.5 else 1.0 - pred
                st.metric("Prediction", f"{label}", delta=f"{confidence*100:.2f}% confidence")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("Notes: keep an eye on memory limits when training in the cloud. For production inference, save models to S3 and load at startup.")
