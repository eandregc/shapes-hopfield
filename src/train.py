from model import HopfieldNetwork
from data import get_training_data, to_ascii

def main():
    patrones = get_training_data()
    datos = list(patrones.values())
    tamano = len(datos[0])

    red = HopfieldNetwork(size=tamano)
    print(f"Entrenando la red con {len(datos)} figuras de {tamano} neuronas...")
    for nombre, patron in patrones.items():
        print(f"\n{nombre}:\n{to_ascii(patron)}")
    red.train(datos)

    red.save_weights("pesos.npy")
    print("\nEntrenamiento finalizado. Pesos guardados en 'pesos.npy'.")

if __name__ == "__main__":
    main()
