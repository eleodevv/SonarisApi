#!/usr/bin/env bash
set -e

pip install -r requirements.txt

echo "[Build] Entrenando MLP..."
python train_mlp.py
echo "[Build] Listo."
