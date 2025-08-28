import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import joblib


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "genre_model_fold4.keras")  
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "labelencoder.pkl")


MAX_PAD_LEN = 174   
N_MFCC = 40         

def extract_features(file_path, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC):
    """
    Loads an audio file and extracts MFCC features.
    Pads or trims the MFCC to a fixed number of time steps.
    Returns the transposed MFCC with shape (time_steps, n_mfcc).
    """
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc.T
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def predict_genre(file_path):
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    print(f"Loading label encoder from {ENCODER_PATH}...")
    label_encoder = joblib.load(ENCODER_PATH)
    print("Label encoder loaded.")

    print(f"Extracting features from: {file_path}")
    features = extract_features(file_path, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC)
    if features is None:
        print("Feature extraction failed.")
        return

    features = np.expand_dims(features, axis=-1)  
    features = np.expand_dims(features, axis=0)    

    prediction = model.predict(features)[0]
    predicted_index = np.argmax(prediction)
    predicted_genre = label_encoder.inverse_transform([predicted_index])[0]
    print(f"Predicted Genre: {predicted_genre}")

if __name__ == "__main__":
    

    predict_genre(os.path.join(os.path.dirname(__file__), "..", "data", "genres_original", "country", "country.00097.wav"))
