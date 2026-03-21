"""Definiciones de acordes organizados por nivel de dificultad."""

ACORDES_BASICOS = {
    'A':  ['A', 'C#', 'E'],
    'Am': ['A', 'C', 'E'],
    'C':  ['C', 'E', 'G'],
    'D':  ['D', 'F#', 'A'],
    'E':  ['E', 'G#', 'B'],
    'Em': ['E', 'G', 'B'],
    'G':  ['G', 'B', 'D'],
    'Dm': ['D', 'F', 'A'],
    'C7': ['C', 'E', 'G', 'A#'],
    'G7': ['G', 'B', 'D', 'F'],
}

ACORDES_MEDIOS = {
    'F':      ['F', 'A', 'C'],
    'Bm':     ['B', 'D', 'F#'],
    'A7':     ['A', 'C#', 'E', 'G'],
    'E7':     ['E', 'G#', 'B', 'D'],
    'Am7':    ['A', 'C', 'E', 'G'],
    'Cmaj7':  ['C', 'E', 'G', 'B'],
    'Dsus4':  ['D', 'G', 'A'],
    'Asus4':  ['A', 'D', 'E'],
}

ACORDES_AVANZADOS = {
    'Gm':  ['G', 'A#', 'D'],
    'F#m': ['F#', 'A', 'C#'],
}

CHORD_DEFINITIONS = {**ACORDES_BASICOS, **ACORDES_MEDIOS, **ACORDES_AVANZADOS}
