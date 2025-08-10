import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import librosa
import os
import tempfile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set page config
st.set_page_config(page_title="Human Voice Classification & Clustering", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

# Load model and scaler
with open("voice_gender_classifier_all_features.pkl", 'rb') as f:
    model = joblib.load(f)

with open("scaler.pkl", 'rb') as f:
    scaler = joblib.load(f)

# Sidebar menu
menu = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Classification", "Clustering"])

if menu == "Introduction":
    st.title("🎙️ Human Voice Classification and Clustering")
    st.markdown("---")
    st.header("🧩 Introduction")
    st.write("""
    This project aims to explore how human voice characteristics can be analyzed using machine learning to **classify gender** and **group similar voice patterns** using **clustering techniques**.
    """)
    st.header("❗ Problem Statement")
    st.write("""
    There is a need for a **lightweight, feature-based ML system** that:
    - Classifies a voice sample's gender.
    - Clusters unlabeled voices into meaningful groups.
    - Is easy to deploy in web apps.
    """)
    st.header("💡 Proposed Solution")
    st.write("""
    1. **Classification** using SVM
    2. **Clustering** using K-Means, DBSCAN, etc.
    """)
    st.header("🛠️ Technologies Used")
    st.table({
        "Component": ["Language", "Data Analysis", "Visualization", "ML", "Interface", "Deployment"],
        "Technology": ["Python", "pandas, NumPy", "Matplotlib, Seaborn", "scikit-learn", "Streamlit", "Pickle"]
    })

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("1. Gender Class Distribution")
    gender_counts = df['label'].value_counts().rename(index={0: 'Female', 1: 'Male'})
    st.bar_chart(gender_counts)

    st.subheader("2. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("3. Feature Distributions by Gender")
    df['gender'] = df['label'].map({0: 'Female', 1: 'Male'})
    features = ['mean_pitch', 'zero_crossing_rate', 'rms_energy', 'log_energy', 'mfcc_1_mean']
    for feature in features:
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x=feature, hue='gender', fill=True, ax=ax)
        st.pyplot(fig)

elif menu == "Classification":
    st.title("🤖 Voice Gender Classification using SVM")

    all_features = list(df.drop(columns=["label"]).columns)

    st.subheader("🎛️ Enter Feature Values Manually")
    input_data = []
    for col in all_features:
        val = st.slider(
            label=col,
            min_value=float(df[col].min()),
            max_value=float(df[col].max()),
            value=float(df[col].mean()),
            step=0.01
        )
        input_data.append(val)

    if st.button("Predict from Manual Input"):
        try:
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)
            label = "👨 Male" if prediction[0] == 1 else "👩 Female"
            st.success(f"Predicted Gender: **{label}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.subheader("📂 Upload CSV to Predict Gender")
    uploaded_file = st.file_uploader("Upload a CSV file with required features", type=["csv"], key="csv")
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            missing = [col for col in all_features if col not in uploaded_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                input_df = uploaded_df[all_features]
                input_scaled = scaler.transform(input_df)
                predictions = model.predict(input_scaled)
                labels = ['👩 Female' if p == 0 else '👨 Male' for p in predictions]
                uploaded_df['Predicted Gender'] = labels
                st.success("✅ Prediction completed!")
                st.dataframe(uploaded_df[['Predicted Gender'] + all_features])

                # Download prediction CSV
                csv = uploaded_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions as CSV", data=csv, file_name="gender_predictions.csv", mime='text/csv')
        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.subheader("🎤 Upload Audio File")
    audio_files = st.file_uploader("Upload one or more WAV audio files", type=["wav"], accept_multiple_files=True, key="audio")
    if audio_files:
        result_data = []
        for audio_file in audio_files:
            try:
                y, sr = librosa.load(audio_file, sr=None)
                features_dict = {}
                for col in all_features:
                    if col.startswith('mfcc_'):
                        mfcc_idx = int(col.split('_')[1])
                        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        if mfcc_idx < mfccs.shape[0]:
                            features_dict[col] = np.mean(mfccs[mfcc_idx])
                        else:
                            features_dict[col] = 0.0
                    elif col == 'mean_pitch':
                        features_dict[col] = np.mean(librosa.yin(y, fmin=50, fmax=300))
                    elif col == 'std_pitch':
                        features_dict[col] = np.std(librosa.yin(y, fmin=50, fmax=300))
                    elif col == 'zero_crossing_rate':
                        features_dict[col] = np.mean(librosa.feature.zero_crossing_rate(y))
                    elif col == 'rms_energy':
                        features_dict[col] = np.mean(librosa.feature.rms(y=y))
                    elif col == 'log_energy':
                        features_dict[col] = np.log(np.mean(librosa.feature.rms(y=y)) + 1e-6)
                    elif col == 'mean_spectral_centroid':
                        features_dict[col] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    else:
                        features_dict[col] = 0.0
                input_df = pd.DataFrame([features_dict])
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                label = "👨 Male" if prediction == 1 else "👩 Female"
                features_dict['Predicted Gender'] = label
                features_dict['Filename'] = audio_file.name
                result_data.append(features_dict)
            except Exception as e:
                st.error(f"Failed to process {audio_file.name}: {e}")

        if result_data:
            result_df = pd.DataFrame(result_data)
            st.success("✅ Audio Predictions Completed")
            st.dataframe(result_df[['Filename', 'Predicted Gender'] + [f for f in all_features if f in result_df.columns]])

            # Download CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Audio Predictions as CSV", data=csv, file_name="audio_gender_predictions.csv", mime='text/csv')

elif menu == "Clustering":
    st.title("🔍 Voice Clustering Analysis")

    # Separate features and labels
    features = df.drop(columns=["label"])
    true_labels = df["label"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Dictionary to store clustering results
    cluster_outputs = {}

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_score = silhouette_score(X_scaled, kmeans_labels)
    cluster_outputs["KMeans"] = {"labels": kmeans_labels, "score": kmeans_score}

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_score = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if np.any(dbscan_labels != -1) else -1
    cluster_outputs["DBSCAN"] = {"labels": dbscan_labels, "score": dbscan_score}

    # Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=2)
    agg_labels = agg.fit_predict(X_scaled)
    agg_score = silhouette_score(X_scaled, agg_labels)
    cluster_outputs["Agglomerative"] = {"labels": agg_labels, "score": agg_score}

    # GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_score = silhouette_score(X_scaled, gmm_labels)
    cluster_outputs["GMM"] = {"labels": gmm_labels, "score": gmm_score}

    st.subheader("📊 Clustering Silhouette Scores")
    score_df = pd.DataFrame({
        "Model": list(cluster_outputs.keys()),
        "Silhouette Score": [v["score"] for v in cluster_outputs.values()]
    }).sort_values(by="Silhouette Score", ascending=False)
    st.dataframe(score_df)

    st.subheader("📈 PCA Clustering Visualizations (Compared to True Gender)")

    for name, result in cluster_outputs.items():
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Clustering result
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=result["labels"], cmap="coolwarm", alpha=0.7)
        ax[0].set_title(f"{name} Clustering")

        # True label visualization
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap="coolwarm", alpha=0.7)
        ax[1].set_title("Actual Gender Labels")

        st.pyplot(fig)
