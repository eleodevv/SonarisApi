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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    """Carga modelos pre-entrenados generados en el build step."""
    loaded = _load_nb_models()
    if not loaded:
        print("[Startup] ADVERTENCIA: Ningún modelo disponible. Ejecuta build.sh primero.")
    yield

app = FastAPI(
    title="Sonaris API",
    description="API para detección de acordes de guitarra usando DSP",
    version="1.0.0",
    lifespan=lifespan
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

# ── Modelos ML: Bayes + MLP ───────────────────────────────────────────────
_NB_MODEL   = None
_NB_ENCODER = None
_NB_FEATURES = None
_MLP_MODEL  = None
_MLP_SCALER = None

def _load_nb_models():
    global _NB_MODEL, _NB_ENCODER, _NB_FEATURES, _MLP_MODEL, _MLP_SCALER
    base = os.path.dirname(os.path.abspath(__file__))
    chroma   = [f'chroma_{n}' for n in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']]
    spectral = ['spectral_centroid','spectral_rolloff','spectral_bandwidth','zero_crossing_rate','rms_energy']
    freq     = [f'freq_pico_{i}' for i in range(1,6)]
    mag      = [f'magnitud_pico_{i}' for i in range(1,6)]
    pitch    = ['pitch_medio','pitch_std','pitch_min','pitch_max']
    _NB_FEATURES = chroma + spectral + freq + mag + pitch

    encoder_path = os.path.join(base, 'label_encoder.pkl')
    mlp_path     = os.path.join(base, 'modelo_mlp.pkl')
    scaler_path  = os.path.join(base, 'scaler_mlp.pkl')

    print(f"[Models] Buscando pkl en: {base}")
    print(f"[Models] modelo_mlp.pkl existe: {os.path.exists(mlp_path)}")
    print(f"[Models] scaler_mlp.pkl existe: {os.path.exists(scaler_path)}")
    print(f"[Models] label_encoder.pkl existe: {os.path.exists(encoder_path)}")
    print(f"[Models] Archivos en directorio: {os.listdir(base)}")

    # Intentar cargar MLP primero (más preciso)
    mlp_path    = os.path.join(base, 'modelo_mlp.pkl')
    scaler_path = os.path.join(base, 'scaler_mlp.pkl')
    if os.path.exists(mlp_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
        try:
            _MLP_MODEL  = joblib.load(mlp_path)
            _MLP_SCALER = joblib.load(scaler_path)
            _NB_ENCODER = joblib.load(encoder_path)
            print("[Models] MLP cargado correctamente.")
        except Exception as e:
            print(f"[Models] MLP pkl incompatible, ignorando: {e}")
            _MLP_MODEL = None

    # Cargar Bayes como fallback
    bayes_path = os.path.join(base, 'modelo_bayes.pkl')
    if os.path.exists(bayes_path) and os.path.exists(encoder_path):
        try:
            _NB_MODEL   = joblib.load(bayes_path)
            if _NB_ENCODER is None:
                _NB_ENCODER = joblib.load(encoder_path)
            print("[Models] Bayes cargado correctamente.")
        except Exception as e:
            print(f"[Models] Bayes pkl incompatible, ignorando: {e}")
            _NB_MODEL = None

    return _MLP_MODEL is not None or _NB_MODEL is not None


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

    # Garantizar mínimo 1 segundo de audio (pad con ceros si es muy corto)
    min_samples = sr  # 1 segundo
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)))

    # ── Picos FFT ──────────────────────────────────────────────────────────
    n_fft = min(16384, len(y))
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

    # n_fft adaptativo para librosa (potencia de 2, máx 2048, no mayor que señal)
    lib_n_fft = 512
    while lib_n_fft * 2 <= min(2048, len(y)):
        lib_n_fft *= 2

    # ── Spectral features via librosa ──────────────────────────────────────
    centroid   = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=lib_n_fft)))
    rolloff    = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=lib_n_fft)))
    bandwidth  = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=lib_n_fft)))
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
    if _NB_MODEL is None and _MLP_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
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

        # Usar MLP si está disponible, sino Bayes
        if _MLP_MODEL is not None and _MLP_SCALER is not None:
            x_scaled = _MLP_SCALER.transform(x)
            proba    = _MLP_MODEL.predict_proba(x_scaled)[0]
            metodo   = "mlp"
        else:
            proba  = _NB_MODEL.predict_proba(x)[0]
            metodo = "naive_bayes"
        classes = _NB_ENCODER.classes_

        # Top 5
        top5_idx = np.argsort(proba)[::-1][:5]
        top5 = [{"acorde": classes[i], "probabilidad": round(float(proba[i]) * 100, 1)} for i in top5_idx]

        print(f"[Bayes] Predicho: {classes[top5_idx[0]]} ({proba[top5_idx[0]]:.2%}) | Top3: {[(classes[i], f'{proba[i]:.2%}') for i in top5_idx[:3]]}")

        return {
            "success": True,
            "acorde_predicho": classes[top5_idx[0]],
            "confianza": round(float(proba[top5_idx[0]]) * 100, 1),
            "top5": top5,
            "metodo": metodo
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al clasificar: {str(e)}")


# ── Entrenamiento personalizado ───────────────────────────────────────────
ACORDES_VALIDOS = ['A', 'Am', 'C', 'D', 'F', 'Bm7']
BASE_DIR        = Path(os.path.dirname(os.path.abspath(__file__)))
CSV_USUARIO     = BASE_DIR / 'dataset_usuario.csv'
CSV_ORIGINAL    = BASE_DIR / 'dataset_real.csv'


@app.get("/samples/descargar")
async def descargar_samples():
    """Descarga el dataset_usuario.csv para guardarlo en el repo"""
    from fastapi.responses import FileResponse
    if not CSV_USUARIO.exists():
        raise HTTPException(status_code=404, detail="No hay samples todavía.")
    return FileResponse(
        path=str(CSV_USUARIO),
        media_type='text/csv',
        filename='dataset_usuario.csv',
    )


@app.post("/samples")
async def subir_sample(audio: UploadFile = File(...), acorde: str = ""):
    """Recibe un audio etiquetado, extrae features y lo guarda en dataset_usuario.csv"""
    acorde = acorde.strip()
    if acorde not in ACORDES_VALIDOS:
        raise HTTPException(status_code=400, detail=f"Acorde inválido. Válidos: {ACORDES_VALIDOS}")

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

        feats['acorde'] = acorde

        import pandas as pd
        fila = pd.DataFrame([feats])

        if CSV_USUARIO.exists():
            fila.to_csv(CSV_USUARIO, mode='a', header=False, index=False)
        else:
            fila.to_csv(CSV_USUARIO, index=False)

        # Contar samples por acorde
        df = pd.read_csv(CSV_USUARIO)
        conteo = df['acorde'].value_counts().to_dict()
        total  = len(df)

        print(f"[Samples] +1 {acorde} | Total: {total} | {conteo}")
        return {"success": True, "acorde": acorde, "total": total, "por_acorde": conteo}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando sample: {str(e)}")


@app.get("/samples/estado")
async def estado_samples():
    """Retorna cuántos samples hay por acorde en el dataset del usuario"""
    import pandas as pd
    if not CSV_USUARIO.exists():
        return {"total": 0, "por_acorde": {a: 0 for a in ACORDES_VALIDOS}, "listo_para_entrenar": False}

    df = pd.read_csv(CSV_USUARIO)
    conteo = {a: int(df[df['acorde'] == a].shape[0]) for a in ACORDES_VALIDOS}
    total  = len(df)
    minimo = min(conteo.values())

    return {
        "total": total,
        "por_acorde": conteo,
        "minimo_por_acorde": minimo,
        "listo_para_entrenar": minimo >= 10,
    }


@app.post("/reentrenar")
async def reentrenar():
    """Combina dataset original + usuario y reentrena el MLP"""
    global _MLP_MODEL, _MLP_SCALER, _NB_ENCODER

    import pandas as pd
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    if not CSV_USUARIO.exists():
        raise HTTPException(status_code=400, detail="No hay samples del usuario todavía.")

    df_usuario  = pd.read_csv(CSV_USUARIO)
    minimo      = df_usuario['acorde'].value_counts().min()
    if minimo < 5:
        raise HTTPException(status_code=400, detail=f"Necesitas al menos 5 samples por acorde. Mínimo actual: {minimo}")

    print(f"[Reentrenar] Samples usuario: {len(df_usuario)}")

    # Combinar con dataset original si existe
    if CSV_ORIGINAL.exists():
        df_original = pd.read_csv(CSV_ORIGINAL)
        # Dar más peso al usuario repitiendo sus samples x5
        df_usuario_boost = pd.concat([df_usuario] * 5, ignore_index=True)
        df = pd.concat([df_original, df_usuario_boost], ignore_index=True)
    else:
        df = df_usuario

    print(f"[Reentrenar] Dataset total: {len(df)} filas")

    CHROMA   = [f'chroma_{n}' for n in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']]
    SPECTRAL = ['spectral_centroid','spectral_rolloff','spectral_bandwidth','zero_crossing_rate','rms_energy']
    FREQ     = [f'freq_pico_{i}' for i in range(1, 6)]
    MAG      = [f'magnitud_pico_{i}' for i in range(1, 6)]
    PITCH    = ['pitch_medio','pitch_std','pitch_min','pitch_max']
    COLS     = CHROMA + SPECTRAL + FREQ + MAG + PITCH

    X  = df[COLS].fillna(0).values
    le = LabelEncoder()
    y  = le.fit_transform(df['acorde'].values)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu', solver='adam',
        max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15, verbose=False,
    )
    mlp.fit(X_train, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"[Reentrenar] Accuracy: {acc:.2%}")

    # Guardar y recargar en memoria
    joblib.dump(mlp,    BASE_DIR / 'modelo_mlp.pkl')
    joblib.dump(scaler, BASE_DIR / 'scaler_mlp.pkl')
    joblib.dump(le,     BASE_DIR / 'label_encoder.pkl')

    _MLP_MODEL  = mlp
    _MLP_SCALER = scaler
    _NB_ENCODER = le

    return {
        "success":  True,
        "accuracy": round(acc * 100, 2),
        "total_samples": len(df),
        "samples_usuario": len(df_usuario),
        "mensaje": f"Modelo reentrenado con {acc:.1%} de precisión",
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
