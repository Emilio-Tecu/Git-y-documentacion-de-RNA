# %load network.py

"""
network.py
~~~~~~~~~~
En este código se implementa una red neuronal para clasificar números hecha desde cero.
Se utiliza 'class' que es un comando utilizado en programación orientado
a objetos y define las propiedades y funciones de un objeto.
La clase network contiene:
    - Inicialización de pesos y bias.
    - Propagación hacia adelante (feedfowrward).
    - Entrenamiento empleando SGD.
    - Y calculo de gradientes por backpropagation.
"""

#### Librerías
import random
import numpy as np

class Network(object):
# Usamos __init__ para configurar la red, definir capas, pesos, bias, etc.
    def __init__(self, sizes):
        """
        sizes: lista que contiene el número de neuronas por capa.
        Ejemplo: [784, 30, 10] significa:
        - 784 neuronas de entrada (imagen 28x28)
        - 30 neuronas en la capa oculta
        - 10 neuronas de salida (dígitos 0-9)

        self.num_layers = número total de capas
        self.sizes = lista de tamaños
        self.biases = lista de vectores columna con los sesgos (bias)
        self.weights = lista de matrices de pesos
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # Los bias y los weights inician con valores aleatorios con una distribución Gaussiana.
        # No se considera la primera capa de bias porque son los datos de entrada.
        # Se tiene un vector de biases y una matriz de weights.

    def feedforward(self, a):
        """
        Calcula la salida de la red para una entrada 'a'.
        Proceso:
        1. Se multiplican pesos * entrada y se suman bias
        2. Se aplica la función de activación sigmoide
        3. Se repite para cada capa
        Retorna: vector de activaciones de la última capa
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    # La función sigmoide se usa para tener continuidad en la transición de 0 a 1.

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Entrena la red usando el algoritmo de Descenso de Gradiente Estocástico (SGD).
        Parámetros:
        -training_data: datos de entrenamiento.
        -epochs: número de épocas (vueltas completas a los datos).
        -mini_batch_size: tamaño de cada sub-lote de entrenamiento.
        -eta: tasa de aprendizaje.
        -test_data: datos de prueba opcionales para monitorear precisión."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Va actualizando los pesos y biases de la red usando backpropagation
        y recibe un mini_batch de datos, para ajustar los parámetros
        en dirección opuesta al gradiente."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                    for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Aplica el algoritmo de backpropagation.
        Retorna:
        - gradientes de pesos
        - gradientes de biases
        Sirve para calcular cómo ajustar cada parámetro de la red
        para minimizar el error de salida."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Este es el feedforward.
        activation = x
        activations = [x] # Lista para guardar todas las activaciones, capa por capa.
        zs = [] # Lista para guardar todos los vectores z, capa por capa.
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = activations[-1] - y # <-- Aquí se implementa la función Cross-entropy.
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Evalúa la presición de la red en datos de prueba.
        Regresa el número de aciertos hechos por la red."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        r"""Regresa el vector de derivadas parciales de la función de costo."""
        return (output_activations-y)


#### Funciones auxiliares.
def sigmoid(z):
    """Función de activación sigmoide."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la función sigmoide"""
    return sigmoid(z)*(1-sigmoid(z))
# Cada que utilicemos una función de activación, necesitamos la 
# función misma y su derivada.