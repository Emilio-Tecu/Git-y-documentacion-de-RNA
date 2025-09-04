# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
Este archivo carga la base de datos MNIST de dígitos escritos a mano.
Convierte las imágenes a vectores de 784 entradas y normaliza.
Se divide en: 
    - training_data (para entrenar la red)
    - validation_data (para validar mientras se entrena)
    - test_data (para evaluar precisión final)
"""

#### Librerías
import pickle
import gzip
import numpy as np
import os

def load_data():
    """
    Devuelve los datos de MNIST como una tupla que contiene los
    datos de entrenamiento, los datos de validación y los datos de
    prueba. Los ``datos_de_entrenamiento`` se devuelven como una
    tupla con dos entradas. La primera entrada contiene las 
    imágenes de entrenamiento reales. Esta es un ndarray de numpy 
    con 50,000 entradas. Cada entrada es, a su vez, un ndarray de 
    numpy con 784 valores, que representan los 28 * 28 = 784 
    píxeles en una sola imagen de MNIST. La segunda entrada de 
    la tupla ``datos_de_entrenamiento`` es un ndarray de numpy 
    que contiene 50,000 entradas. Esas entradas son solo los 
    valores de dígito (0...9) para las imágenes correspondientes 
    contenidas en la primera entrada de la tupla. 
    Los ``datos_de_validación`` y ``datos_de_prueba`` son 
    similares, excepto que cada uno contiene solo 10,000 imágenes. 
    Este es un buen formato de datos, pero para su uso en 
    redes neuronales es útil modificar un poco el formato de los 
    ``datos_de_entrenamiento``. 
    Esto se hace en la función envoltorio ``load_data_wrapper()``.
    """
    ruta_mnist = os.path.join(os.path.dirname(__file__), 'mnist.pkl.gz')
    f = gzip.open(ruta_mnist, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Devuelve una tupla que contiene ``(datos_de_entrenamiento, 
    datos_de_validación, datos_de_prueba)``. Basado en 
    ``cargar_datos``, pero el formato es más conveniente para su 
    uso en nuestra implementación de redes neuronales. 
    En particular, ``datos_de_entrenamiento`` es una lista que 
    contiene 50,000 2-tuplas ``(x, y)``. ``x`` es un numpy.ndarray 
    de 784 dimensiones que contiene la imagen de entrada. ``y`` 
    es un numpy.ndarray de 10 dimensiones que representa el vector 
    unitario correspondiente al dígito correcto para ``x``. 
    ``datos_de_validación`` y ``datos_de_prueba`` son listas que 
    contienen 10,000 2-tuplas ``(x, y)``. En cada caso, ``x`` es 
    un numpy.ndarray de 784 dimensiones que contiene la imagen de 
    entrada, y ``y`` es la clasificación correspondiente, es decir, 
    los valores de dígitos (enteros) correspondientes a ``x``. 
    Obviamente, esto significa que estamos utilizando formatos 
    ligeramente diferentes para los datos de entrenamiento y 
    los datos de validación / prueba. Estos formatos resultan 
    ser los más convenientes para su uso en nuestro código de red 
    neuronal."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Convierte una etiqueta entera (ej. 3) a un vector columna
    con un 1 en la posición correspondiente.
    Ejemplo: para '3' → [0,0,0,1,0,0,0,0,0,0]^T"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
