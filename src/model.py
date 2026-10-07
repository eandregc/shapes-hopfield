import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, data):
        for pattern in data:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.size

    def energy(self, state):
        return -0.5 * state @ self.weights @ state

    def predict(self, pattern, steps=10):
        state = np.copy(pattern)
        for _ in range(steps):
            anterior = np.copy(state)
            for i in range(self.size):
                suma = np.dot(self.weights[i], state)
                state[i] = 1 if suma >= 0 else -1
            if np.array_equal(state, anterior):
                break
        return state

    def recognize(self, pattern, patrones):
        """Recupera el patrón y lo compara con las figuras conocidas.
        Devuelve el nombre de la figura más parecida y el patrón recuperado."""
        recuperado = self.predict(pattern)
        mejor, mejor_similitud = None, -1
        for nombre, original in patrones.items():
            similitud = np.mean(recuperado == original)
            if similitud > mejor_similitud:
                mejor, mejor_similitud = nombre, similitud
        return mejor, mejor_similitud, recuperado

    def save_weights(self, filepath):
        np.save(filepath, self.weights)

    def load_weights(self, filepath):
        self.weights = np.load(filepath)
