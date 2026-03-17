"""
Entrena MLP desde dataset_real.csv.
Se llama en startup de main.py para evitar incompatibilidades de versión.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

BASE = Path(os.path.dirname(os.path.abspath(__file__)))

CHROMA_COLS   = [f'chroma_{n}' for n in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']]
SPECTRAL_COLS = ['spectral_centroid','spectral_rolloff','spectral_bandwidth','zero_crossing_rate','rms_energy']
FREQ_COLS     = [f'freq_pico_{i}' for i in range(1, 6)]
MAG_COLS      = [f'magnitud_pico_{i}' for i in range(1, 6)]
PITCH_COLS    = ['pitch_medio','pitch_std','pitch_min','pitch_max']
FEATURE_COLS  = CHROMA_COLS + SPECTRAL_COLS + FREQ_COLS + MAG_COLS + PITCH_COLS


def train():
    csv_path = BASE / 'dataset_real.csv'
    if not csv_path.exists():
        print("[MLP] dataset_real.csv no encontrado, saltando entrenamiento MLP.")
        return None, None, None

    print(f"[MLP] Cargando {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[MLP] Filas: {len(df)} | Acordes: {df['acorde'].nunique()} — {sorted(df['acorde'].unique())}")

    X = df[FEATURE_COLS].fillna(0).values
    le = LabelEncoder()
    y  = le.fit_transform(df['acorde'].values)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y)

    print(f"[MLP] Entrenando ({len(X_train)} train / {len(X_test)} test)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )
    mlp.fit(X_train, y_train)

    acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"[MLP] Accuracy: {acc:.2%}")

    joblib.dump(mlp,    BASE / 'modelo_mlp.pkl')
    joblib.dump(scaler, BASE / 'scaler_mlp.pkl')
    joblib.dump(le,     BASE / 'label_encoder.pkl')
    print("[MLP] Modelos guardados.")
    return mlp, scaler, le

if __name__ == '__main__':
    train()
