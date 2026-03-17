"""
Guitar Chord Detection using Digital Signal Processing (DSP)
NO Machine Learning - Pure FFT and Pitch Detection

Author: Sistema de Detección de Acordes
Purpose: Evaluate if a user played the correct guitar chord
"""

import numpy as np
import librosa
from collections import Counter

# ============================================================================
# CHORD DEFINITIONS
# ============================================================================

# ============================================================================
# ACORDES POR NIVEL DE DIFICULTAD
# ============================================================================

# BÁSICOS (10) - Acordes simples, posiciones fáciles, muy comunes
ACORDES_BASICOS = {
    'A': ['A', 'C#', 'E'],      # La mayor - muy fácil
    'Am': ['A', 'C', 'E'],      # La menor - muy fácil
    'C': ['C', 'E', 'G'],       # Do mayor - básico
    'D': ['D', 'F#', 'A'],      # Re mayor - básico
    'E': ['E', 'G#', 'B'],      # Mi mayor - fácil
    'Em': ['E', 'G', 'B'],      # Mi menor - muy fácil
    'G': ['G', 'B', 'D'],       # Sol mayor - común
    'Dm': ['D', 'F', 'A'],      # Re menor - fácil
    'C7': ['C', 'E', 'G', 'A#'], # Do séptima - blues
    'G7': ['G', 'B', 'D', 'F']   # Sol séptima - jazz/blues
}

# MEDIOS (8) - Requieren más práctica, cejillas simples
ACORDES_MEDIOS = {
    'F': ['F', 'A', 'C'],       # Fa mayor - cejilla
    'Bm': ['B', 'D', 'F#'],     # Si menor - cejilla
    'A7': ['A', 'C#', 'E', 'G'], # La séptima
    'E7': ['E', 'G#', 'B', 'D'], # Mi séptima
    'Am7': ['A', 'C', 'E', 'G'], # La menor séptima
    'Cmaj7': ['C', 'E', 'G', 'B'], # Do mayor séptima
    'Dsus4': ['D', 'G', 'A'],    # Re suspendido 4
    'Asus4': ['A', 'D', 'E']     # La suspendido 4
}

# AVANZADOS (2) - Acordes complejos, cejillas completas
ACORDES_AVANZADOS = {
    'Gm': ['G', 'A#', 'D'],      # Sol menor - cejilla en 3er traste
    'F#m': ['F#', 'A', 'C#']     # Fa# menor - cejilla en 2do traste
}

# Diccionario completo (para compatibilidad)
CHORD_DEFINITIONS = {
    **ACORDES_BASICOS,
    **ACORDES_MEDIOS,
    **ACORDES_AVANZADOS
}

# Musical note frequencies (A4 = 440 Hz as reference)
NOTE_FREQUENCIES = {
    'C': [65.41, 130.81, 261.63, 523.25, 1046.50],
    'C#': [69.30, 138.59, 277.18, 554.37, 1108.73],
    'D': [73.42, 146.83, 293.66, 587.33, 1174.66],
    'D#': [77.78, 155.56, 311.13, 622.25, 1244.51],
    'E': [82.41, 164.81, 329.63, 659.25, 1318.51],
    'F': [87.31, 174.61, 349.23, 698.46, 1396.91],
    'F#': [92.50, 185.00, 369.99, 739.99, 1479.98],
    'G': [98.00, 196.00, 392.00, 783.99, 1567.98],
    'G#': [103.83, 207.65, 415.30, 830.61, 1661.22],
    'A': [110.00, 220.00, 440.00, 880.00, 1760.00],
    'A#': [116.54, 233.08, 466.16, 932.33, 1864.66],
    'B': [123.47, 246.94, 493.88, 987.77, 1975.53]
}

# ============================================================================
# FREQUENCY TO NOTE CONVERSION
# ============================================================================

def freq_to_note(freq, tolerance=3.0):
    """
    Convert a frequency (Hz) to the closest musical note name.
    
    Uses the equal temperament formula:
    n = 12 * log2(f / 440)
    
    Parameters:
    -----------
    freq : float
        Frequency in Hz
    tolerance : float
        Maximum difference in semitones to consider a match
    
    Returns:
    --------
    str or None
        Note name (e.g., 'A', 'C#', 'D') or None if no match
    """
    if freq < 60 or freq > 1300:  # Outside guitar range
        return None
    
    # Calculate semitones from A4 (440 Hz)
    # n = 12 * log2(f / 440)
    semitones_from_a4 = 12 * np.log2(freq / 440.0)
    
    # Round to nearest semitone
    nearest_semitone = round(semitones_from_a4)
    
    # Check if within tolerance
    if abs(semitones_from_a4 - nearest_semitone) > tolerance / 12.0:
        return None
    
    # Note names in chromatic scale
    note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    
    # Get note index (A4 is index 0)
    note_index = nearest_semitone % 12
    
    return note_names[note_index]

# ============================================================================
# AUDIO PREPROCESSING
# ============================================================================

def preprocess_audio(y, sr):
    """
    Preprocess audio signal for better note detection.
    
    Parameters:
    -----------
    y : np.array
        Audio time series
    sr : int
        Sample rate
    
    Returns:
    --------
    np.array
        Filtered audio signal
    """
    # Apply bandpass filter for guitar frequencies (80-1200 Hz)
    from scipy import signal as scipy_signal
    
    nyquist = sr / 2
    low = 80 / nyquist
    high = 1200 / nyquist
    
    b, a = scipy_signal.butter(4, [low, high], btype='band')
    y_filtered = scipy_signal.filtfilt(b, a, y)
    
    # Normalize
    y_filtered = y_filtered / (np.max(np.abs(y_filtered)) + 1e-10)
    
    return y_filtered

# ============================================================================
# NOTE DETECTION FROM AUDIO
# ============================================================================

def detect_notes_from_audio(file_path, method='hybrid'):
    """
    Load audio file and detect musical notes using DSP techniques.
    
    Parameters:
    -----------
    file_path : str
        Path to audio file (WAV format recommended)
    method : str
        Detection method: 'fft', 'piptrack', or 'hybrid' (default)
    
    Returns:
    --------
    list
        List of detected note names (e.g., ['D', 'F#', 'A'])
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=22050, duration=3.0)
    
    # Preprocess
    y = preprocess_audio(y, sr)
    
    detected_notes = []
    
    if method == 'fft' or method == 'hybrid':
        # Method 1: FFT-based detection
        notes_fft = _detect_notes_fft(y, sr)
        detected_notes.extend(notes_fft)
    
    if method == 'piptrack' or method == 'hybrid':
        # Method 2: Pitch tracking
        notes_pitch = _detect_notes_piptrack(y, sr)
        detected_notes.extend(notes_pitch)
    
    # Count occurrences and get most common notes
    note_counter = Counter(detected_notes)
    
    # Get notes that appear at least once
    unique_notes = [note for note, count in note_counter.most_common() if count >= 1]
    
    # Return top 4 most common notes (typical for guitar chords)
    return unique_notes[:4]

def _detect_notes_fft(y, sr):
    """
    Detect notes using Fast Fourier Transform (FFT).
    
    Parameters:
    -----------
    y : np.array
        Audio signal
    sr : int
        Sample rate
    
    Returns:
    --------
    list
        Detected notes
    """
    # Apply window function
    windowed = y * np.hanning(len(y))
    
    # Compute FFT with high resolution
    n_fft = max(8192, len(windowed))
    fft = np.fft.rfft(windowed, n=n_fft)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    # Find peaks in magnitude spectrum
    from scipy.signal import find_peaks
    
    threshold = np.max(magnitude) * 0.08  # 8% of max
    peaks, properties = find_peaks(magnitude, height=threshold, distance=5)
    
    # Get frequencies of peaks
    peak_freqs = freqs[peaks]
    peak_mags = magnitude[peaks]
    
    # Filter to guitar range
    valid_mask = (peak_freqs >= 70) & (peak_freqs <= 1300)
    peak_freqs = peak_freqs[valid_mask]
    peak_mags = peak_mags[valid_mask]
    
    # Sort by magnitude and take top 10
    sorted_indices = np.argsort(peak_mags)[::-1]
    top_freqs = peak_freqs[sorted_indices[:10]]
    
    # Convert to notes
    notes = []
    for freq in top_freqs:
        note = freq_to_note(freq, tolerance=4.0)
        if note:
            notes.append(note)
    
    return notes

def _detect_notes_piptrack(y, sr):
    """Detect notes using librosa's pitch tracking."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=70, fmax=1300)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    notes = []
    for pitch in pitch_values:
        note = freq_to_note(pitch, tolerance=3.0)
        if note:
            notes.append(note)
    return notes

def detect_notes_fast(file_path):
    """
    Detección rápida usando solo FFT + soundfile (sin librosa, sin filtros lentos).
    """
    import soundfile as sf
    from scipy.signal import find_peaks
    from collections import Counter

    y, sr = sf.read(file_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    # Tomar solo 1.5 segundos del centro (donde suena el acorde)
    total = len(y)
    start = max(0, total // 4)
    end   = min(total, start + int(sr * 1.5))
    y     = y[start:end]

    # Normalizar
    mx = np.max(np.abs(y))
    if mx > 0:
        y = y / mx

    # FFT resolución balanceada (16384 es suficiente para acordes, 2x más rápido que 32768)
    n_fft    = 16384
    windowed = y * np.hanning(len(y))
    fft_mag  = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs    = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Solo rango guitarra (70-1300 Hz)
    mask     = (freqs >= 70) & (freqs <= 1300)
    fft_mag  = fft_mag[mask]
    freqs    = freqs[mask]

    # Picos
    threshold = np.max(fft_mag) * 0.07
    peaks, _  = find_peaks(fft_mag, height=threshold, distance=6)

    if len(peaks) == 0:
        return []

    peak_freqs = freqs[peaks]
    peak_mags  = fft_mag[peaks]

    # Top 12 por magnitud
    top_idx   = np.argsort(peak_mags)[::-1][:12]
    top_freqs = peak_freqs[top_idx]

    notes   = [freq_to_note(f, tolerance=4.0) for f in top_freqs]
    notes   = [n for n in notes if n]
    counter = Counter(notes)
    return [n for n, _ in counter.most_common(4)]




def check_chord(detected_notes, expected_chord, threshold=0.6):
    """
    Check if detected notes match the expected chord.
    
    Parameters:
    -----------
    detected_notes : list
        List of detected note names
    expected_chord : str
        Expected chord name (e.g., 'D', 'Am', 'C')
    threshold : float
        Minimum match ratio (0.0 to 1.0) to consider correct
    
    Returns:
    --------
    dict
        Result dictionary with:
        - 'match': bool (True if chord matches)
        - 'confidence': float (match percentage)
        - 'expected_notes': list (notes in expected chord)
        - 'detected_notes': list (notes detected)
        - 'matched_notes': list (notes that matched)
        - 'missing_notes': list (expected notes not detected)
    """
    if expected_chord not in CHORD_DEFINITIONS:
        return {
            'match': False,
            'confidence': 0.0,
            'error': f"Unknown chord: {expected_chord}"
        }
    
    expected_notes = CHORD_DEFINITIONS[expected_chord]
    
    # Find matched notes
    matched_notes = [note for note in expected_notes if note in detected_notes]
    
    # Find missing notes
    missing_notes = [note for note in expected_notes if note not in detected_notes]
    
    # Calculate match ratio
    match_ratio = len(matched_notes) / len(expected_notes)
    
    # Penalize if extra notes detected (not in chord)
    extra_notes = [note for note in detected_notes if note not in expected_notes]
    if extra_notes:
        penalty = len(extra_notes) * 0.1
        match_ratio = max(0, match_ratio - penalty)
    
    # Determine if match
    is_match = match_ratio >= threshold
    
    return {
        'match': is_match,
        'confidence': match_ratio * 100,
        'expected_notes': expected_notes,
        'detected_notes': detected_notes,
        'matched_notes': matched_notes,
        'missing_notes': missing_notes,
        'extra_notes': extra_notes
    }

# ============================================================================
# MAIN EXAMPLE
# ============================================================================

def main():
    """
    Example usage of the chord detection system.
    """
    print("="*60)
    print("GUITAR CHORD DETECTION - DSP ONLY (NO ML)")
    print("="*60)
    print()
    
    # Example 1: Check a specific audio file
    print("Example 1: Detect chord from audio file")
    print("-" * 60)
    
    # Simulated example (replace with actual file path)
    audio_file = "user_play.wav"
    expected_chord = "D"
    
    print(f"Expected chord: {expected_chord}")
    print(f"Expected notes: {CHORD_DEFINITIONS[expected_chord]}")
    print(f"Audio file: {audio_file}")
    print()
    
    try:
        # Detect notes from audio
        detected_notes = detect_notes_from_audio(audio_file)
        print(f"Detected notes: {detected_notes}")
        print()
        
        # Check if chord matches
        result = check_chord(detected_notes, expected_chord)
        
        print("Result:")
        print(f"  Match: {'✓ YES' if result['match'] else '✗ NO'}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Matched notes: {result['matched_notes']}")
        print(f"  Missing notes: {result['missing_notes']}")
        print(f"  Extra notes: {result['extra_notes']}")
        
    except FileNotFoundError:
        print(f"⚠ File not found: {audio_file}")
        print("Please provide a valid audio file path.")
    
    print()
    print("="*60)
    
    # Example 2: Test frequency to note conversion
    print("\nExample 2: Frequency to Note Conversion")
    print("-" * 60)
    
    test_frequencies = [146.83, 293.66, 369.99, 440.00, 587.33]
    print("Test frequencies:")
    for freq in test_frequencies:
        note = freq_to_note(freq)
        print(f"  {freq:.2f} Hz → {note}")
    
    print()
    print("="*60)

if __name__ == "__main__":
    main()
