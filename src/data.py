import numpy as np

# Cada figura se representa en una cuadrícula de 10x10 (100 neuronas).
# El '#' corresponde a un píxel activo (1) y el '.' a un píxel inactivo (-1).

CUADRADO = [
    "..........",
    ".########.",
    ".########.",
    ".##....##.",
    ".##....##.",
    ".##....##.",
    ".##....##.",
    ".########.",
    ".########.",
    "..........",
]

TRIANGULO = [
    "....##....",
    "....##....",
    "...####...",
    "...####...",
    "..##..##..",
    "..##..##..",
    ".##....##.",
    ".##....##.",
    "##########",
    "##########",
]

CRUZ = [
    "....##....",
    "....##....",
    "....##....",
    "....##....",
    "##########",
    "##########",
    "....##....",
    "....##....",
    "....##....",
    "....##....",
]

CIRCULO = [
    "...####...",
    "..##..##..",
    ".##....##.",
    "##......##",
    "##......##",
    "##......##",
    "##......##",
    ".##....##.",
    "..##..##..",
    "...####...",
]

FIGURAS = {
    "Cuadrado": CUADRADO,
    "Triangulo": TRIANGULO,
    "Cruz": CRUZ,
    "Circulo": CIRCULO,
}

def to_bipolar(figura):
    """Convierte la figura ASCII a un vector bipolar (1 / -1) de tamaño 100."""
    return np.array([1 if c == "#" else -1 for fila in figura for c in fila])

def to_ascii(vector, lado=10):
    """Convierte un vector bipolar a su representación ASCII de 'lado' x 'lado'."""
    return "\n".join(
        "".join("#" if v == 1 else "." for v in vector[i * lado:(i + 1) * lado])
        for i in range(lado)
    )

def add_noise(vector, porcentaje=0.2, seed=None):
    """Invierte aleatoriamente un porcentaje de los píxeles para simular ruido."""
    rng = np.random.default_rng(seed)
    ruidoso = np.copy(vector)
    n_ruido = int(len(vector) * porcentaje)
    indices = rng.choice(len(vector), n_ruido, replace=False)
    ruidoso[indices] *= -1
    return ruidoso

def get_training_data():
    return {nombre: to_bipolar(figura) for nombre, figura in FIGURAS.items()}

def get_test_data(porcentaje=0.2, seed=42):
    patrones = get_training_data()
    pruebas = []
    for i, (nombre, original) in enumerate(patrones.items()):
        pruebas.append({
            "nombre": nombre,
            "original": original,
            "noisy": add_noise(original, porcentaje, seed=seed + i),
        })
    return pruebas
