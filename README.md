## Music Genre Classification (CNN + BiLSTM)

A deep learning project that classifies music into 10 genres from the GTZAN dataset using MFCC features and a Keras CNN + Bidirectional LSTM model. Includes CLI inference and a Streamlit UI with explanations.

### Project Structure

- `src/`: training and inference
  - `main.py`: train with 5-fold cross-validation, saves models to `models/`
  - `predict.py`: predict a genre for a `.wav` file using a saved model
  - `webapp.py`: Streamlit app for interactive predictions and visualizations
- `data/genres_original/`: GTZAN audio organized by genre (10 folders)
- `models/`: saved models `genre_model_fold*.keras` and `labelencoder.pkl` (ignored by git)
- `docs/readme.md`: detailed project report

### Requirements

- Python 3.10+
- Recommended: create a virtual environment

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install core libs:

```bash
pip install tensorflow==2.* librosa scikit-learn joblib streamlit matplotlib numpy
```

### Dataset

- GTZAN (10 genres; 30s `.wav` clips). Place as:
```
data/genres_original/<genre>/<file>.wav
```

### Model Overview

- Input features: MFCCs with shape `(174, 40, 1)`
- Architecture (from `src/main.py`):
  - Conv2D → MaxPooling2D → BatchNorm → Dropout (x2)
  - `TimeDistributed(Flatten())`
  - Bidirectional LSTM(64) → Dropout
  - Dense softmax over 10 classes
- Training: Adam, categorical crossentropy, early stopping, 5-fold CV
- Artifacts saved to `models/genre_model_fold{1..5}.keras` and `models/labelencoder.pkl`

### Training

```bash
python src/main.py
```

This will:
- Extract MFCCs from `data/genres_original`
- Encode labels and save `models/labelencoder.pkl`
- Train 5 folds and save each as `models/genre_model_foldX.keras`

### Inference (CLI)

`src/predict.py` loads `models/genre_model_fold4.keras` by default. Edit `MODEL_PATH` to change the model.

```bash
python src/predict.py
# or
python -c "from src.predict import predict_genre; print(predict_genre('path/to/file.wav'))"
```

### Streamlit App

Run the interactive UI:

```bash
streamlit run src/webapp.py
```

Features:
- Upload a `.wav` file and select a fold model
- View waveform, log-mel spectrogram
- See top-3 predicted genres and per-frame importance (occlusion)

### Model Artifacts and Git

- Models and checkpoints can be large; they are ignored by git. Make sure `.gitignore` contains entries like:

```
models/
*.keras
*.h5
*.hdf5
*.pkl
*.joblib
*.ckpt
```

If you accidentally committed a model, untrack it after updating `.gitignore`:

```bash
git rm -r --cached models/
git commit -m "Stop tracking model artifacts"
```

### Notes

- Validation accuracy may be lower than training due to overfitting; consider augmentation, stronger regularization, or ensembling all folds.
- For reproduction, ensure consistent `librosa` and `tensorflow` versions.


