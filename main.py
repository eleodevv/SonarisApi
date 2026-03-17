"""
FastAPI Backend para Detección de Acordes
Sistema DSP sin Machine Learning
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import os
import joblib

from detector_acordes_dsp import (  # type: ignore
    detect_notes_fast,
    check_chord,
    CHORD_DEFINITIONS,
    ACORDES_BASICOS,
    ACORDES_MEDIOS,
    ACORDES_AVANZADOS
)

app = FastAPI(
    title="Sonaris API",
    description="API para detección de acordes de guitarra usando DSP",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Sonaris API - Detección de Acordes",
        "version": "1.0.0",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """Verificar estado del servidor"""
    return {"status": "healthy"}

@app.get("/acordes")
async def listar_acordes():
    """Lista todos los acordes disponibles organizados por nivel"""
    return {
        "total": len(CHORD_DEFINITIONS),
        "acordes": {
            "basicos": list(ACORDES_BASICOS.keys()),
            "medios": list(ACORDES_MEDIOS.keys()),
            "avanzados": list(ACORDES_AVANZADOS.keys())
        },
        "definiciones": CHORD_DEFINITIONS
    }

@app.get("/acorde/{nombre}")
async def obtener_acorde(nombre: str):
    """Obtiene información de un acorde específico"""
    nombre = nombre.upper()
    
    if nombre not in CHORD_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Acorde '{nombre}' no encontrado")
    
    # Determinar nivel
    if nombre in ACORDES_BASICOS:
        nivel = "basico"
    elif nombre in ACORDES_MEDIOS:
        nivel = "medio"
    else:
        nivel = "avanzado"
    
    return {
        "acorde": nombre,
        "notas": CHORD_DEFINITIONS[nombre],
        "nivel": nivel,
        "num_notas": len(CHORD_DEFINITIONS[nombre])
    }

@app.post("/detectar")
async def detectar_acorde(
    audio: UploadFile = File(...),
    acorde_esperado: str = None
):
    # FastAPI a veces no parsea form fields junto con files en multipart
    # El acorde puede venir en el filename como fallback
    if not acorde_esperado and audio.filename:
        # Flutter puede enviar el acorde en el filename: "A.wav", "Am.wav"
        name = audio.filename.replace('.wav','').replace('.mp3','').strip()
        if name in CHORD_DEFINITIONS:
            acorde_esperado = name
    """
    Detecta el acorde de un archivo de audio
    
    Parameters:
    - audio: Archivo de audio (WAV, MP3, etc.)
    - acorde_esperado: (Opcional) Acorde que se espera detectar
    """
    
    # Validar tipo de archivo (Flutter puede enviar octet-stream)
    allowed = ['audio/', 'application/octet-stream', 'video/']
    ct = audio.content_type or ''
    if ct and not any(ct.startswith(a) for a in allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {ct}"
        )
    
    try:
        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Detectar notas usando DSP rápido
        detected_notes = detect_notes_fast(temp_path)
        print(f"[DSP] Acorde esperado: {acorde_esperado} | Detectadas: {detected_notes}")
        
        # Si no se detectaron notas
        if not detected_notes:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "No se detectaron notas claras",
                    "notas_detectadas": [],
                    "acorde_detectado": None,
                    "confianza": 0.0
                }
            )
        
        # Encontrar el mejor acorde
        mejor_acorde = None
        mejor_confianza = 0.0
        
        for nombre_acorde, notas_acorde in CHORD_DEFINITIONS.items():
            matched = sum(1 for nota in notas_acorde if nota in detected_notes)
            confianza = matched / len(notas_acorde)
            
            # Penalizar notas extra
            extra = sum(1 for nota in detected_notes if nota not in notas_acorde)
            confianza = max(0, confianza - extra * 0.1)
            
            # Bonus si coincide con el esperado
            if acorde_esperado and nombre_acorde.upper() == acorde_esperado.upper():
                confianza *= 1.2
            
            if confianza > mejor_confianza:
                mejor_confianza = confianza
                mejor_acorde = nombre_acorde
        
        # Verificar si es correcto (si hay acorde esperado)
        es_correcto = None
        if acorde_esperado:
            acorde_esperado = acorde_esperado.upper()
            if acorde_esperado in CHORD_DEFINITIONS:
                result = check_chord(detected_notes, acorde_esperado, threshold=0.5)
                es_correcto = result['match']
        
        # Determinar nivel del acorde detectado
        if mejor_acorde in ACORDES_BASICOS:
            nivel = "basico"
        elif mejor_acorde in ACORDES_MEDIOS:
            nivel = "medio"
        else:
            nivel = "avanzado"
        
        # Limpiar archivo temporal
        Path(temp_path).unlink()
        
        return {
            "success": True,
            "acorde_detectado": mejor_acorde,
            "confianza": round(mejor_confianza * 100, 1),
            "notas_detectadas": detected_notes[:5],
            "notas_esperadas": CHORD_DEFINITIONS[mejor_acorde],
            "nivel": nivel,
            "es_correcto": es_correcto,
            "acorde_esperado": acorde_esperado if acorde_esperado else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar audio: {str(e)}"
        )

@app.post("/verificar")
async def verificar_acorde(
    audio: UploadFile = File(...),
    acorde_esperado: str = None
):
    """
    Verifica si el audio corresponde al acorde esperado
    Similar a /detectar pero enfocado en validación
    """
    # Fallback: leer el acorde del filename si el form field no llegó
    if not acorde_esperado and audio.filename:
        name = audio.filename.replace('.wav','').replace('.mp3','').strip()
        if name.upper() in CHORD_DEFINITIONS or name in CHORD_DEFINITIONS:
            acorde_esperado = name

    if not acorde_esperado:
        raise HTTPException(status_code=400, detail="Debe proporcionar el acorde esperado")

    # Normalizar: strip + buscar case-insensitive
    acorde_esperado = acorde_esperado.strip()
    # Buscar match case-insensitive en las definiciones
    match = next((k for k in CHORD_DEFINITIONS if k.upper() == acorde_esperado.upper()), None)
    if not match:
        # Intentar con el valor tal cual (para F#m, Am7, etc.)
        match = next((k for k in CHORD_DEFINITIONS if k == acorde_esperado), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Acorde '{acorde_esperado}' no encontrado. Disponibles: {list(CHORD_DEFINITIONS.keys())}")
    acorde_esperado = match
    
    try:
        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Detectar notas
        detected_notes = detect_notes_fast(temp_path)
        print(f"[DSP] Verificar: {acorde_esperado} | Detectadas: {detected_notes}")
        
        # Verificar acorde
        result = check_chord(detected_notes, acorde_esperado, threshold=0.6)
        
        # Limpiar
        Path(temp_path).unlink()
        
        return {
            "success": True,
            "acorde_esperado": acorde_esperado,
            "es_correcto": result['match'],
            "confianza": round(result['confidence'], 1),
            "notas_esperadas": result['expected_notes'],
            "notas_detectadas": result['detected_notes'],
            "notas_correctas": result['matched_notes'],
            "notas_faltantes": result['missing_notes'],
            "notas_extra": result['extra_notes']
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al verificar acorde: {str(e)}"
        )

# ── Naive Bayes: cargar modelos al inicio ──────────────────────────────────
_NB_MODEL = None
_NB_ENCODER = None
_NB_FEATURES = None

def _load_nb_models():
    global _NB_MODEL, _NB_ENCODER, _NB_FEATURES
    base = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join(base, 'modelo_bayes.pkl')
    encoder_path = os.path.join(base, 'label_encoder.pkl')
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        _NB_MODEL   = joblib.load(model_path)
        _NB_ENCODER = joblib.load(encoder_path)
        # Features en el mismo orden que train_bayes.py
        chroma  = [f'chroma_{n}' for n in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']]
        spectral = ['spectral_centroid','spectral_rolloff','spectral_bandwidth','zero_crossing_rate','rms_energy']
        freq    = [f'freq_pico_{i}' for i in range(1,6)]
        mag     = [f'magnitud_pico_{i}' for i in range(1,6)]
        pitch   = ['pitch_medio','pitch_std','pitch_min','pitch_max']
        _NB_FEATURES = chroma + spectral + freq + mag + pitch
        return True
    return False

_load_nb_models()


def _extract_features_for_bayes(file_path: str) -> dict:
    """Extrae las mismas features que el dataset para pasarlas al modelo Bayes."""
    import soundfile as sf
    from scipy.signal import find_peaks

    y, sr = sf.read(file_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    # Normalizar
    mx = np.max(np.abs(y))
    if mx > 0:
        y = y / mx

    # ── Picos FFT ──────────────────────────────────────────────────────────
    n_fft = 16384
    windowed = y * np.hanning(len(y))
    fft_mag  = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs    = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask     = (freqs >= 70) & (freqs <= 1300)
    fft_sub  = fft_mag[mask]
    freq_sub = freqs[mask]
    threshold = np.max(fft_sub) * 0.07 if len(fft_sub) > 0 else 0
    peaks, _ = find_peaks(fft_sub, height=threshold, distance=6)
    top5 = np.argsort(fft_sub[peaks])[::-1][:5] if len(peaks) >= 5 else np.argsort(fft_sub[peaks])[::-1]
    peak_f = freq_sub[peaks[top5]] if len(peaks) > 0 else np.zeros(5)
    peak_m = fft_sub[peaks[top5]] if len(peaks) > 0 else np.zeros(5)
    # Pad to 5
    pf = np.zeros(5); pm = np.zeros(5)
    pf[:len(peak_f)] = peak_f; pm[:len(peak_m)] = peak_m

    # ── Spectral features via librosa ──────────────────────────────────────
    centroid   = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff    = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    bandwidth  = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zcr        = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms        = float(np.mean(librosa.feature.rms(y=y)))

    # ── Chroma ────────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # shape (12,)
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    # ── Pitch ─────────────────────────────────────────────────────────────
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.max(magnitudes) * 0.1]
    pitch_vals = pitch_vals[pitch_vals > 0]
    if len(pitch_vals) == 0:
        pitch_vals = np.array([0.0])

    feat = {}
    for i, n in enumerate(note_names):
        feat[f'chroma_{n}'] = float(chroma_mean[i])
    feat['spectral_centroid']  = centroid
    feat['spectral_rolloff']   = rolloff
    feat['spectral_bandwidth'] = bandwidth
    feat['zero_crossing_rate'] = zcr
    feat['rms_energy']         = rms
    for i in range(5):
        feat[f'freq_pico_{i+1}']     = float(pf[i])
        feat[f'magnitud_pico_{i+1}'] = float(pm[i])
    feat['pitch_medio'] = float(np.mean(pitch_vals))
    feat['pitch_std']   = float(np.std(pitch_vals))
    feat['pitch_min']   = float(np.min(pitch_vals))
    feat['pitch_max']   = float(np.max(pitch_vals))
    return feat


@app.post("/clasificar")
async def clasificar_acorde_bayes(audio: UploadFile = File(...)):
    """
    Clasifica el acorde usando Naive Bayes entrenado con el dataset DSP.
    Retorna el acorde predicho y las probabilidades de los top-5 acordes.
    """
    if _NB_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo Bayes no disponible. Ejecuta train_bayes.py primero."
        )

    allowed = ['audio/', 'application/octet-stream', 'video/']
    ct = audio.content_type or ''
    if ct and not any(ct.startswith(a) for a in allowed):
        raise HTTPException(status_code=400, detail=f"Tipo no soportado: {ct}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        feats = _extract_features_for_bayes(tmp_path)
        Path(tmp_path).unlink()

        # Vector de features en el orden correcto
        x = np.array([[feats.get(f, 0.0) for f in _NB_FEATURES]])

        proba = _NB_MODEL.predict_proba(x)[0]
        classes = _NB_ENCODER.classes_

        # Top 5
        top5_idx = np.argsort(proba)[::-1][:5]
        top5 = [{"acorde": classes[i], "probabilidad": round(float(proba[i]) * 100, 1)} for i in top5_idx]

        return {
            "success": True,
            "acorde_predicho": classes[top5_idx[0]],
            "confianza": round(float(proba[top5_idx[0]]) * 100, 1),
            "top5": top5,
            "metodo": "naive_bayes"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al clasificar: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
