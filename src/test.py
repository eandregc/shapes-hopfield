import numpy as np
from model import HopfieldNetwork
from data import get_training_data, get_test_data, to_ascii

def main():
    patrones = get_training_data()
    datos_prueba = get_test_data(porcentaje=0.2)
    tamano = len(datos_prueba[0]["noisy"])

    red = HopfieldNetwork(size=tamano)
    red.load_weights("pesos.npy")

    print("Iniciando prueba de reconocimiento de figuras con ruido...")
    aciertos = 0
    for prueba in datos_prueba:
        nombre, similitud, recuperado = red.recognize(prueba["noisy"], patrones)

        print(f"\n{'=' * 40}")
        print(f"Figura esperada: {prueba['nombre']}")
        print(f"\nFigura Original:\n{to_ascii(prueba['original'])}")
        print(f"\nFigura Ruidosa:\n{to_ascii(prueba['noisy'])}")
        print(f"\nFigura Recuperada:\n{to_ascii(recuperado)}")

        exito = np.array_equal(recuperado, prueba["original"])
        aciertos += exito
        print(f"\nFigura reconocida: {nombre} (similitud {similitud:.0%})")
        print(f"Recuperación exitosa?: {exito}")

    print(f"\n{'=' * 40}")
    print(f"Figuras reconocidas correctamente: {aciertos}/{len(datos_prueba)}")

if __name__ == "__main__":
    main()
