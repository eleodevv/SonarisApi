"""
Entrena una CNN con mel-spectrogramas desde dataset_augmented/.
Estructura esperada:
    dataset_augmented/
        basico/A/*.wav
        basico/Am/*.wav
        medio/F/*.wav
        avanzado/Bm7/*.wav
        ...
Genera: modelo_cnn.keras, label_classes.npy
"""

import os
import numpy as np
import librosa
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# ── Configuración ─────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset_augmented"
OUTPUT_DIR  = Path(__file__).resolve().parent

SR          = 22050   # sample rate
DURATION    = 2.0     # segundos por muestra
N_MELS      = 128     # bandas mel
HOP_LENGTH  = 512
N_FFT       = 2048
IMG_H       = 128     # alto del espectrograma (n_mels)
IMG_W       = 87      # ancho (frames para 2s a sr=22050, hop=512)
EPOCHS      = 40
BATCH_SIZE  = 32


# ── Carga de datos ────────────────────────────────────────────────────────────

def wav_to_melspec(path: str) -> np.ndarray:
    """Convierte un WAV a mel-spectrogram normalizado (IMG_H x IMG_W)."""
    y, sr = librosa.load(path, sr=SR, duration=DURATION, mono=True)
    # Padding si el audio es más corto que DURATION
    target = int(SR * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Redimensionar al tamaño fijo
    if mel_db.shape[1] != IMG_W:
        mel_db = tf.image.resize(mel_db[..., np.newaxis], [IMG_H, IMG_W]).numpy()[..., 0]

    # Normalizar entre 0 y 1
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def load_dataset():
    X, y, classes = [], [], []
    label_map = {}

    for nivel_dir in sorted(DATASET_DIR.iterdir()):
        if not nivel_dir.is_dir():
            continue
        for acorde_dir in sorted(nivel_dir.iterdir()):
            if not acorde_dir.is_dir():
                continue
            acorde = acorde_dir.name
            if acorde not in label_map:
                label_map[acorde] = len(label_map)
                classes.append(acorde)

            wavs = list(acorde_dir.glob("*.wav"))
            print(f"  {acorde}: {len(wavs)} archivos")
            for wav in wavs:
                try:
                    spec = wav_to_melspec(str(wav))
                    X.append(spec)
                    y.append(label_map[acorde])
                except Exception as e:
                    print(f"    [skip] {wav.name}: {e}")

    X = np.array(X)[..., np.newaxis]  # (N, H, W, 1) — canal para Conv2D
    y = np.array(y)
    return X, y, np.array(classes)


# ── Modelo CNN ────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(IMG_H, IMG_W, 1)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train():
    print(f"[CNN] Cargando dataset desde {DATASET_DIR}...")
    X, y, classes = load_dataset()
    print(f"[CNN] Total: {len(X)} muestras | Clases: {list(classes)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)

    model = build_model(num_classes=len(classes))
    model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[CNN] Test accuracy: {acc:.2%} | Loss: {loss:.4f}")

    model.save(OUTPUT_DIR / "modelo_cnn.keras")
    np.save(OUTPUT_DIR / "label_classes.npy", classes)
    print("[CNN] Modelo guardado: modelo_cnn.keras + label_classes.npy")


if __name__ == "__main__":
    train()
