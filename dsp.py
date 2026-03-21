"""Detección de notas y verificación de acordes usando DSP (FFT)."""

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
from collections import Counter
from chords import CHORD_DEFINITIONS

NOTE_NAMES = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


def _freq_to_note(freq: float, tolerance: float = 4.0) -> str | None:
    if freq < 60 or freq > 1300:
        return None
    semitones = 12 * np.log2(freq / 440.0)
    nearest   = round(semitones)
    if abs(semitones - nearest) > tolerance / 12.0:
        return None
    return NOTE_NAMES[nearest % 12]


def detect_notes_fast(file_path: str) -> list[str]:
    """Detecta las notas presentes en un archivo WAV usando FFT."""
    y, sr = sf.read(file_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    # Tomar 1.5 s del centro del audio
    start = max(0, len(y) // 4)
    y     = y[start: start + int(sr * 1.5)]

    mx = np.max(np.abs(y))
    if mx > 0:
        y = y / mx

    windowed = y * np.hanning(len(y))
    fft_mag  = np.abs(np.fft.rfft(windowed, n=16384))
    freqs    = np.fft.rfftfreq(16384, 1.0 / sr)

    mask    = (freqs >= 70) & (freqs <= 1300)
    fft_mag = fft_mag[mask]
    freqs   = freqs[mask]

    peaks, _ = find_peaks(fft_mag, height=np.max(fft_mag) * 0.07, distance=6)
    if len(peaks) == 0:
        return []

    top_freqs = freqs[peaks][np.argsort(fft_mag[peaks])[::-1][:12]]
    notes     = [_freq_to_note(f) for f in top_freqs]
    counter   = Counter(n for n in notes if n)
    return [n for n, _ in counter.most_common(4)]


def check_chord(detected_notes: list, expected_chord: str, threshold: float = 0.6) -> dict:
    """Verifica si las notas detectadas corresponden al acorde esperado."""
    if expected_chord not in CHORD_DEFINITIONS:
        return {'match': False, 'confidence': 0.0, 'error': f"Acorde desconocido: {expected_chord}"}

    expected      = CHORD_DEFINITIONS[expected_chord]
    matched       = [n for n in expected if n in detected_notes]
    missing       = [n for n in expected if n not in detected_notes]
    extra         = [n for n in detected_notes if n not in expected]
    ratio         = max(0.0, len(matched) / len(expected) - len(extra) * 0.1)

    return {
        'match':          ratio >= threshold,
        'confidence':     ratio * 100,
        'expected_notes': expected,
        'detected_notes': detected_notes,
        'matched_notes':  matched,
        'missing_notes':  missing,
        'extra_notes':    extra,
    }
