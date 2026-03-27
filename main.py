"""
Sonaris API - Detección de Acordes de Guitarra
FastAPI + CNN (TensorFlow/Keras) + DSP (FFT)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile, os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks

from chords import CHORD_DEFINITIONS, ACORDES_BASICOS, ACORDES_MEDIOS, ACORDES_AVANZADOS
from dsp import detect_notes_fast, check_chord

# ── Modelo ────────────────────────────────────────────────────────────────────
_CNN_MODEL = None
_CLASSES   = None

SR         = 22050
DURATION   = 2.0
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048
IMG_H      = 128
IMG_W      = 87


def _load_models():
    global _CNN_MODEL, _CLASSES
    base = Path(os.path.dirname(os.path.abspath(__file__)))
    model_path   = base / "training" / "modelo_cnn.keras"
    classes_path = base / "training" / "label_classes.npy"
    try:
        import tensorflow as tf
        _CNN_MODEL = tf.keras.models.load_model(str(model_path))
        _CLASSES   = np.load(str(classes_path), allow_pickle=True)
        print(f"[Models] CNN cargado. Clases: {list(_CLASSES)}")
        return True
    except Exception as e:
        print(f"[Models] Error cargando CNN: {e}")
        return False


@asynccontextmanager
async def lifespan(app):
    if not _load_models():
        print("[Startup] Modelo no disponible. Ejecuta training/train_cnn.py primero.")
    yield


app = FastAPI(title="Sonaris API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── Helpers ───────────────────────────────────────────────────────────────────

def _nivel(acorde: str) -> str:
    if acorde in ACORDES_BASICOS: return "basico"
    if acorde in ACORDES_MEDIOS:  return "medio"
    return "avanzado"


def _validate_audio(audio: UploadFile):
    ct = audio.content_type or ''
    if ct and not any(ct.startswith(p) for p in ['audio/', 'application/octet-stream', 'video/']):
        raise HTTPException(status_code=400, detail=f"Tipo no soportado: {ct}")


def _save_temp(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(content)
        return tmp.name


def _wav_to_melspec(file_path: str) -> np.ndarray:
    """Convierte WAV a mel-spectrogram normalizado listo para la CNN."""
    import tensorflow as tf
    y, sr = librosa.load(file_path, sr=SR, duration=DURATION, mono=True)
    target = int(SR * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] != IMG_W:
        mel_db = tf.image.resize(mel_db[..., np.newaxis], [IMG_H, IMG_W]).numpy()[..., 0]

    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Sonaris API", "version": "3.0.0", "status": "online"}

@app.get("/health")
@app.head("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/acordes")
async def listar_acordes():
    return {
        "total": len(CHORD_DEFINITIONS),
        "acordes": {
            "basicos":   list(ACORDES_BASICOS.keys()),
            "medios":    list(ACORDES_MEDIOS.keys()),
            "avanzados": list(ACORDES_AVANZADOS.keys()),
        },
        "definiciones": CHORD_DEFINITIONS,
    }

@app.get("/acorde/{nombre}")
async def obtener_acorde(nombre: str):
    nombre = nombre.upper()
    if nombre not in CHORD_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Acorde '{nombre}' no encontrado")
    return {"acorde": nombre, "notas": CHORD_DEFINITIONS[nombre],
            "nivel": _nivel(nombre), "num_notas": len(CHORD_DEFINITIONS[nombre])}


@app.post("/detectar")
async def detectar_acorde(audio: UploadFile = File(...), acorde_esperado: str = None):
    """Detecta el acorde usando DSP (FFT)."""
    _validate_audio(audio)
    if not acorde_esperado and audio.filename:
        name = audio.filename.replace('.wav', '').replace('.mp3', '').strip()
        if name in CHORD_DEFINITIONS:
            acorde_esperado = name
    try:
        tmp   = _save_temp(await audio.read())
        notes = detect_notes_fast(tmp)
        Path(tmp).unlink()

        if not notes:
            return JSONResponse(content={"success": False, "message": "No se detectaron notas",
                                         "notas_detectadas": [], "acorde_detectado": None, "confianza": 0.0})

        mejor, conf = None, 0.0
        for nombre_acorde, notas_acorde in CHORD_DEFINITIONS.items():
            matched = sum(1 for n in notas_acorde if n in notes)
            extra   = sum(1 for n in notes if n not in notas_acorde)
            c = max(0, matched / len(notas_acorde) - extra * 0.1)
            if acorde_esperado and nombre_acorde.upper() == acorde_esperado.upper():
                c *= 1.2
            if c > conf:
                conf, mejor = c, nombre_acorde

        es_correcto = None
        if acorde_esperado:
            acorde_esperado = acorde_esperado.upper()
            if acorde_esperado in CHORD_DEFINITIONS:
                es_correcto = check_chord(notes, acorde_esperado, threshold=0.5)['match']

        return {"success": True, "acorde_detectado": mejor, "confianza": round(conf * 100, 1),
                "notas_detectadas": notes[:5], "notas_esperadas": CHORD_DEFINITIONS[mejor],
                "nivel": _nivel(mejor), "es_correcto": es_correcto, "acorde_esperado": acorde_esperado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verificar")
async def verificar_acorde(audio: UploadFile = File(...), acorde_esperado: str = None):
    """Verifica si el audio corresponde al acorde esperado usando DSP."""
    _validate_audio(audio)
    if not acorde_esperado and audio.filename:
        name = audio.filename.replace('.wav', '').replace('.mp3', '').strip()
        if name.upper() in CHORD_DEFINITIONS:
            acorde_esperado = name
    if not acorde_esperado:
        raise HTTPException(status_code=400, detail="Debe proporcionar el acorde esperado")

    match = next((k for k in CHORD_DEFINITIONS if k.upper() == acorde_esperado.strip().upper()), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Acorde '{acorde_esperado}' no encontrado")

    try:
        tmp   = _save_temp(await audio.read())
        notes = detect_notes_fast(tmp)
        Path(tmp).unlink()
        result = check_chord(notes, match, threshold=0.6)
        return {"success": True, "acorde_esperado": match, "es_correcto": result['match'],
                "confianza": round(result['confidence'], 1), "notas_esperadas": result['expected_notes'],
                "notas_detectadas": result['detected_notes'], "notas_correctas": result['matched_notes'],
                "notas_faltantes": result['missing_notes'], "notas_extra": result['extra_notes']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clasificar")
async def clasificar_acorde(audio: UploadFile = File(...)):
    """Clasifica el acorde usando la CNN con mel-spectrogramas."""
    if _CNN_MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecuta training/train_cnn.py primero.")
    _validate_audio(audio)
    try:
        tmp  = _save_temp(await audio.read())
        spec = _wav_to_melspec(tmp)
        Path(tmp).unlink()

        proba    = _CNN_MODEL.predict(spec, verbose=0)[0]
        top5_idx = np.argsort(proba)[::-1][:5]

        return {
            "success":         True,
            "acorde_predicho": str(_CLASSES[top5_idx[0]]),
            "confianza":       round(float(proba[top5_idx[0]]) * 100, 1),
            "top5":            [{"acorde": str(_CLASSES[i]), "probabilidad": round(float(proba[i]) * 100, 1)}
                                 for i in top5_idx],
            "metodo":          "cnn",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
