# 🎸 Sonaris API

API REST para detección de acordes de guitarra usando procesamiento digital de señales (DSP).

## 🚀 Instalación

```bash
cd api
pip install -r requirements.txt
```

## ▶️ Ejecutar

```bash
python main.py
```

O con uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 Endpoints

### GET `/`
Información básica de la API

### GET `/health`
Verificar estado del servidor

### GET `/acordes`
Lista todos los acordes disponibles organizados por nivel

**Respuesta:**
```json
{
  "total": 20,
  "acordes": {
    "basicos": ["A", "Am", "C", "D", ...],
    "medios": ["F", "Bm", "A7", ...],
    "avanzados": ["Gm", "F#m"]
  },
  "definiciones": {
    "A": ["A", "C#", "E"],
    ...
  }
}
```

### GET `/acorde/{nombre}`
Obtiene información de un acorde específico

**Ejemplo:** `/acorde/A`

**Respuesta:**
```json
{
  "acorde": "A",
  "notas": ["A", "C#", "E"],
  "nivel": "basico",
  "num_notas": 3
}
```

### POST `/detectar`
Detecta el acorde de un archivo de audio

**Parámetros:**
- `audio` (file): Archivo de audio (WAV, MP3, etc.)
- `acorde_esperado` (string, opcional): Acorde que se espera detectar

**Respuesta:**
```json
{
  "success": true,
  "acorde_detectado": "A",
  "confianza": 85.5,
  "notas_detectadas": ["A", "C#", "E"],
  "notas_esperadas": ["A", "C#", "E"],
  "nivel": "basico",
  "es_correcto": true,
  "acorde_esperado": "A"
}
```

### POST `/verificar`
Verifica si el audio corresponde al acorde esperado

**Parámetros:**
- `audio` (file): Archivo de audio
- `acorde_esperado` (string, requerido): Acorde a verificar

**Respuesta:**
```json
{
  "success": true,
  "acorde_esperado": "A",
  "es_correcto": true,
  "confianza": 85.5,
  "notas_esperadas": ["A", "C#", "E"],
  "notas_detectadas": ["A", "C#", "E"],
  "notas_correctas": ["A", "C#", "E"],
  "notas_faltantes": [],
  "notas_extra": []
}
```

## 🧪 Probar con cURL

```bash
# Listar acordes
curl http://localhost:8000/acordes

# Obtener info de un acorde
curl http://localhost:8000/acorde/A

# Detectar acorde
curl -X POST http://localhost:8000/detectar \
  -F "audio=@audio.wav"

# Verificar acorde
curl -X POST http://localhost:8000/verificar \
  -F "audio=@audio.wav" \
  -F "acorde_esperado=A"
```

## 📱 Integración con Flutter

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> detectarAcorde(String audioPath) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://localhost:8000/detectar'),
  );
  
  request.files.add(await http.MultipartFile.fromPath('audio', audioPath));
  
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  
  return json.decode(responseData);
}
```

## 🐳 Docker (Opcional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t sonaris-api .
docker run -p 8000:8000 sonaris-api
```

## 📚 Documentación Interactiva

Una vez ejecutando, visita:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
