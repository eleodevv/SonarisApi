#!/usr/bin/env bash
set -e

pip install -r requirements.txt

echo "[Build] Entrenando MLP..."
python train_mlp.py
echo "[Build] Listo."

echo "[Build] Verificando pkl generados:"
ls -lh modelo_mlp.pkl scaler_mlp.pkl label_encoder.pkl 2>&1 || echo "[Build] ERROR: pkl no encontrados"
