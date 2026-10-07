# Reconocimiento de Figuras con Red de Hopfield
Una red de Hopfield es un tipo de red neuronal artificial recurrente que funciona como un sistema de memoria asociativa. En este proyecto se utiliza para **reconocer figuras geométricas** (cuadrado, triángulo, cruz y círculo) dibujadas en una cuadrícula de 10x10 píxeles. La red almacena las figuras limpias y, al recibir una versión incompleta o con ruido, recupera la figura original e identifica de cuál se trata.

**Características clave:**

* **Arquitectura:** Está completamente conectada. Las 100 neuronas (una por píxel) se conectan entre sí, pero ninguna está conectada consigo misma.
* **Pesos simétricos:** La fuerza de conexión de la neurona A hacia la neurona B es exactamente la misma que de la B hacia la A. Se calculan con la regla de Hebb a partir de las figuras de entrenamiento.
* **Valores:** Cada píxel se representa de forma bipolar: `1` para un píxel activo (`#`) y `-1` para un píxel inactivo (`.`).
* **Dinámica de energía:** La red evoluciona iterativamente minimizando una "función de energía". Cada cambio de estado de las neuronas hace que la red descienda hacia un valle de energía (mínimo local). Cuando la red se estabiliza y deja de cambiar, ha "recordado" o reconstruido la figura.
* **Reconocimiento:** Una vez recuperado el patrón, se compara con las figuras almacenadas y se reporta la más parecida junto con su porcentaje de similitud.
* **Capacidad limitada:** Si se intenta almacenar un número excesivo de figuras en relación al tamaño de la red (aprox. 0.138·N), o si las figuras son muy parecidas entre sí, los estados se mezclan y la red comienza a recordar patrones falsos (estados espurios).

## Figuras almacenadas

```
Cuadrado      Triangulo     Cruz          Circulo
..........    ....##....    ....##....    ...####...
.########.    ....##....    ....##....    ..##..##..
.########.    ...####...    ....##....    .##....##.
.##....##.    ...####...    ....##....    ##......##
.##....##.    ..##..##..    ##########    ##......##
.##....##.    ..##..##..    ##########    ##......##
.##....##.    .##....##.    ....##....    ##......##
.########.    .##....##.    ....##....    .##....##.
.########.    ##########    ....##....    ..##..##..
..........    ##########    ....##....    ...####...
```

## Distribución del Proyecto
shapes_hopfield/
├── src/
│   ├── data.py       # Define las figuras limpias (ASCII), su conversión a vectores bipolares y la generación de ruido.
│   ├── model.py      # Contiene la lógica matemática y la clase HopfieldNetwork (entrenamiento, recuperación y reconocimiento).
│   ├── train.py      # Script que entrena el modelo con las figuras y exporta los pesos a 'pesos.npy'.
│   └── test.py       # Script que importa los pesos, agrega ruido a cada figura y evalúa su reconocimiento.
├── .gitignore        # Excluye la caché de Python y el archivo binario de pesos al subir a Git.
└── requirements.txt  # Dependencias del entorno virtual (numpy).

## Ejecución

```bash
pip install -r requirements.txt
cd src
python train.py   # Entrena la red y genera pesos.npy
python test.py    # Agrega 20% de ruido a cada figura y la reconoce
```
