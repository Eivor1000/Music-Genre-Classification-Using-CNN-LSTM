import os
import io
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import joblib


MAX_PAD_LEN = 174
N_MFCC = 40
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# Quiet TensorFlow/oneDNN chatter and deprecation noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel(logging.ERROR)


def load_model(model_filename: str):
    model_path = os.path.join(MODELS_DIR, model_filename)
    # Avoid stale graphs across reruns and avoid compile overhead/state
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def load_label_encoder():
    enc_path = os.path.join(MODELS_DIR, "labelencoder.pkl")
    return joblib.load(enc_path)


@st.cache_data(show_spinner=False)
def load_audio(file_bytes: bytes, sr_target: int = 22050):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr_target, mono=True, duration=30)
    return y, sr


def extract_mfcc(y: np.ndarray, sr: int, max_pad_len: int = MAX_PAD_LEN, n_mfcc: int = N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]
    # (time, n_mfcc)
    return mfcc.T


def predict(model: tf.keras.Model, encoder, mfcc_time_major: np.ndarray):
    x = np.expand_dims(mfcc_time_major, axis=-1)  # (T, C, 1)
    x = np.expand_dims(x, axis=0)  # (1, T, C, 1)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = encoder.inverse_transform([idx])[0]
    return label, probs


def occlusion_importance(model: tf.keras.Model, base_probs: np.ndarray, mfcc_time_major: np.ndarray, window: int = 8):
    """
    Slide a zero-mask window across time frames and measure drop in top-class prob.
    Returns importance per time step (length = T).
    """
    top_idx = int(np.argmax(base_probs))
    T = mfcc_time_major.shape[0]
    importance = np.zeros(T, dtype=np.float32)

    # Pre-expand once to avoid repeated shape work
    def run_pred(mfcc_tm: np.ndarray):
        x = np.expand_dims(mfcc_tm, axis=-1)
        x = np.expand_dims(x, axis=0)
        return model.predict(x, verbose=0)[0]

    for start in range(0, T):
        end = min(T, start + window)
        masked = mfcc_time_major.copy()
        masked[start:end, :] = 0.0
        probs_masked = run_pred(masked)
        drop = float(base_probs[top_idx] - probs_masked[top_idx])
        # Accumulate drop uniformly across masked frames
        if end > start:
            importance[start:end] += drop / (end - start)

    # Normalize to [0, 1] for display
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance


def plot_waveform(y: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#2c7fb8")
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_logmel(y: np.ndarray, sr: int):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    logS = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(logS, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_mfcc(mfcc_tm: np.ndarray, importance: np.ndarray | None = None, sr: int | None = None):
    # mfcc_tm shape: (T, N_MFCC)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = ax.imshow(mfcc_tm.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("MFCC (time x coeff)")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("MFCC index")
    fig.colorbar(img, ax=ax)

    # Overlay importance as a line (scaled to MFCC coeff axis)
    if importance is not None and len(importance) == mfcc_tm.shape[0]:
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(importance)), importance * (N_MFCC - 1), color="crimson", alpha=0.8)
        ax2.set_ylabel("Importance (scaled)")
        ax2.set_ylim(0, N_MFCC - 1)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Music Genre Classifier", layout="wide")
st.title("Music Genre Classification — Visual + Explainable")

with st.sidebar:
    st.header("Settings")
    # Model selection from available folds
    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.startswith("genre_model_fold") and f.endswith(".keras")])
    selected_model = st.selectbox("Select model fold", model_files, index=min(3, len(model_files) - 1) if model_files else 0)
    window = st.slider("Importance window (frames)", min_value=4, max_value=24, value=8, step=2)

uploaded = st.file_uploader("Upload a WAV file", type=["wav"]) 

if uploaded is not None and selected_model:
    # Simple session cache instead of Streamlit cache to avoid TF name_scope bug
    if "_model_cache" not in st.session_state:
        st.session_state._model_cache = {}
    if selected_model not in st.session_state._model_cache:
        with st.spinner("Loading model..."):
            st.session_state._model_cache[selected_model] = load_model(selected_model)
    model = st.session_state._model_cache[selected_model]
    encoder = load_label_encoder()

    with st.spinner("Reading audio..."):
        y, sr = load_audio(uploaded.getvalue())

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_waveform(y, sr))
        st.pyplot(plot_logmel(y, sr))

    with st.spinner("Extracting MFCCs and predicting..."):
        mfcc_tm = extract_mfcc(y, sr)
        label, probs = predict(model, encoder, mfcc_tm)

    # Top-3 probabilities
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(encoder.inverse_transform([int(i)])[0], float(probs[i])) for i in top3_idx]

    with col2:
        st.subheader("Prediction")
        st.write(f"Predicted genre: **{label}**")
        st.write("Top-3 confidences:")
        for g, p in top3:
            st.write(f"- {g}: {p:.3f}")

    with st.spinner("Computing importance over time (occlusion)..."):
        importance = occlusion_importance(model, probs, mfcc_tm, window=window)

    st.subheader("MFCC with Frame Importance")
    st.pyplot(plot_mfcc(mfcc_tm, importance))

else:
    st.info("Upload a WAV file to see predictions and visualizations.")


