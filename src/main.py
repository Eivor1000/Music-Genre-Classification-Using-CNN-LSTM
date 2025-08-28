import os
import librosa
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, LSTM, Dense, BatchNormalization, TimeDistributed, Flatten, Bidirectional # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import joblib

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "genres_original")
MAX_PAD_LEN = 174      
NUM_MFCC = 40         
K = 5                  


def extract_features(file_path, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=NUM_MFCC)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.T  
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


 


def load_data():
    features, labels = [], []
    print("Extracting features from audio files...")
    for genre in os.listdir(DATASET_PATH):
        genre_path = os.path.join(DATASET_PATH, genre)
        if not os.path.isdir(genre_path):
            continue
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            mfcc = extract_features(file_path)
            if mfcc is not None:
                features.append(mfcc)
                labels.append(genre)
    print("Feature extraction complete.")
    return np.array(features), np.array(labels)


def build_model(input_shape, num_classes):
    print("Building CNN + LSTM model...")
    inp = Input(shape=input_shape)
    

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    

    x = TimeDistributed(Flatten())(x)
    

    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    

    out = Dense(num_classes, activation='softmax')(x)
    

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model built.")
    return model

if __name__ == "__main__":

    X, y = load_data()

    X = X[..., np.newaxis]
    
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = to_categorical(label_encoder.fit_transform(y))
    joblib.dump(label_encoder, os.path.join(os.path.dirname(__file__), "..", "models", "labelencoder.pkl"))
    

    print("Starting K-Fold Cross Validation with LSTM + CNN...")
    kfold = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []

    for train_idx, test_idx in kfold.split(X):
        print(f"\nFold {fold_no}/{K}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        model = build_model(X_train.shape[1:], y_encoded.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=2
        )

        scores = model.evaluate(X_test, y_test, verbose=0)
        print(f"Fold {fold_no} — Loss: {scores[0]:.4f} — Accuracy: {scores[1]:.4f}")
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        model.save(os.path.join(os.path.dirname(__file__), "..", "models", f"genre_model_fold{fold_no}.keras"))
        print(f"Model for fold {fold_no} saved as models/genre_model_fold{fold_no}.keras")
        fold_no += 1

    print("\nAverage Performance over K folds:")
    print(f"Avg Accuracy: {np.mean(acc_per_fold):.4f}")
    print(f"Avg Loss: {np.mean(loss_per_fold):.4f}")
