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
        if name in CHORD_DEFINITIONS:
            acorde_esperado = name

    if not acorde_esperado:
        raise HTTPException(
            status_code=400,
            detail="Debe proporcionar el acorde esperado"
        )
    
    acorde_esperado = acorde_esperado.upper()
    
    if acorde_esperado not in CHORD_DEFINITIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Acorde '{acorde_esperado}' no encontrado"
        )
    
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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
