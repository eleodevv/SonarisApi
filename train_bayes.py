
"""
Entrenamiento de Naive Bayes para clasificación de acordes de guitarra.
Usa el dataset dataset_dsp.csv generado con características DSP reales.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset_dsp.csv')
MODEL_BAYES_PATH = os.path.join(BASE_DIR, 'modelo_bayes.pkl')
MODEL_REGRESION_PATH = os.path.join(BASE_DIR, 'modelo_regresion.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

# ── Features ───────────────────────────────────────────────────────────────
CHROMA_COLS = [f'chroma_{n}' for n in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']]
SPECTRAL_COLS = ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate', 'rms_energy']
FREQ_COLS = ['freq_pico_1', 'freq_pico_2', 'freq_pico_3', 'freq_pico_4', 'freq_pico_5']
MAG_COLS  = ['magnitud_pico_1', 'magnitud_pico_2', 'magnitud_pico_3', 'magnitud_pico_4', 'magnitud_pico_5']
PITCH_COLS = ['pitch_medio', 'pitch_std', 'pitch_min', 'pitch_max']

FEATURE_COLS = CHROMA_COLS + SPECTRAL_COLS + FREQ_COLS + MAG_COLS + PITCH_COLS


def train():
    print("Cargando dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Filas: {len(df)} | Acordes únicos: {df['acorde'].nunique()}")
    print(f"  Acordes: {sorted(df['acorde'].unique())}")

    # Verificar columnas
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ADVERTENCIA: columnas faltantes: {missing}")
        FEATURE_COLS_USED = [c for c in FEATURE_COLS if c in df.columns]
    else:
        FEATURE_COLS_USED = FEATURE_COLS

    X = df[FEATURE_COLS_USED].fillna(0).values
    y_labels = df['acorde'].values

    # Codificar etiquetas
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ── Naive Bayes ────────────────────────────────────────────────────────
    print("\nEntrenando Naive Bayes (GaussianNB)...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Regresión Lineal Múltiple ──────────────────────────────────────────
    # Predice rms_energy (proxy de confianza/volumen) a partir de features espectrales
    print("Entrenando Regresión Lineal Múltiple...")
    reg_features = CHROMA_COLS + SPECTRAL_COLS[:3]  # centroid, rolloff, bandwidth
    reg_features = [c for c in reg_features if c in df.columns]
    X_reg = df[reg_features].fillna(0).values
    y_reg = df['rms_energy'].fillna(0).values

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    lr = LinearRegression()
    lr.fit(X_reg_train, y_reg_train)
    r2 = lr.score(X_reg_test, y_reg_test)
    print(f"  R² Regresión: {r2:.4f}")

    # ── Guardar modelos ────────────────────────────────────────────────────
    joblib.dump(nb, MODEL_BAYES_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump({'model': lr, 'features': reg_features}, MODEL_REGRESION_PATH)
    print(f"\nModelos guardados:")
    print(f"  {MODEL_BAYES_PATH}")
    print(f"  {MODEL_REGRESION_PATH}")
    print(f"  {ENCODER_PATH}")
    print(f"\nFeatures usadas ({len(FEATURE_COLS_USED)}): {FEATURE_COLS_USED}")

    return nb, le, lr, FEATURE_COLS_USED


if __name__ == '__main__':
    train()
