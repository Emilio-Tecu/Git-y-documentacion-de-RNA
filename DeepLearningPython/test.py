"""
CÓDIGO DE PRUEBA PARA RED NEURONAL - CLASIFICACIÓN DE DÍGITOS MNIST
Adaptado para Python 3.5.2

Uso en terminal:
    python test.py

Parámetros de la red neuronal (network.py):
    - 2do parámetro: número de épocas de entrenamiento
    - 3er parámetro: tamaño del lote (batch size)
    - 4to parámetro: tasa de aprendizaje (eta)

Autor original:
    Michał Dobrzański, 2016
    dobrzanski.michal.daniel@gmail.com

Modificado por:
    Emilio
    Repositorio: https://github.com/Emilio-Tecu/Git-y-documentacion-de-RNA.git
"""

# ----------------------
# Carga del conjunto de datos MNIST:

import mnist_loader
# Se carga el dataser MNIST.
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data) # Para compatibilidad con la implementación.

# ---------------------
# Entrenamiento de la red neuronal.
import network

# Se crea la red.
net = network.Network([784, 30, 10])
# Se entrena usando SGD, con 30 épocas, 10 el tamaño del mini-batch.
# Y 3 el learning rate.
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)


# ----------------------
# - network2.py example:
#import network2

'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)
'''

# chapter 3 - Overfitting example - too many epochs of learning applied on small (1k samples) amount od data.
# Overfitting is treating noise as a signal.
'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True)
'''

# chapter 3 - Regularization (weight decay) example 1 (only 1000 of training data and 30 hidden neurons)
'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000], 400, 10, 0.5,
    evaluation_data=test_data,
    lmbda = 0.1, # this is a regularization parameter
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)
'''

# chapter 3 - Early stopping implemented
'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data[:1000], 30, 10, 0.5,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    early_stopping_n=10)
'''

# chapter 4 - The vanishing gradient problem - deep networks are hard to train with simple SGD algorithm
# this network learns much slower than a shallow one.
'''
net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)
'''


# ----------------------
# Theano and CUDA
# ----------------------

"""
    This deep network uses Theano with GPU acceleration support.
    I am using Ubuntu 16.04 with CUDA 7.5.
    Tutorial:
    http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu

    The following command will update only Theano:
        sudo pip install --upgrade --no-deps theano

    The following command will update Theano and Numpy/Scipy (warning bellow):
        sudo pip install --upgrade theano

"""

"""
    Below, there is a testing function to check whether your computations have been made on CPU or GPU.
    If the result is 'Used the cpu' and you want to have it in gpu,     do the following:
    1) install theano:
        sudo python3.5 -m pip install Theano
    2) download and install the latest cuda:
        https://developer.nvidia.com/cuda-downloads
        I had some issues with that, so I followed this idea (better option is to download the 1,1GB package as .run file):
        http://askubuntu.com/questions/760242/how-can-i-force-16-04-to-add-a-repository-even-if-it-isnt-considered-secure-eno
        You may also want to grab the proper NVidia driver, choose it form there:
        System Settings > Software & Updates > Additional Drivers.
    3) should work, run it with:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python3.5 test.py
        http://deeplearning.net/software/theano/tutorial/using_gpu.html
    4) Optionally, you can add cuDNN support from:
        https://developer.nvidia.com/cudnn


"""
# def testTheano():
#     from theano import function, config, shared, sandbox
#     import theano.tensor as T
#     import numpy
#     import time
#     print("Testing Theano library...")
#     vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
#     iters = 1000

#     rng = numpy.random.RandomState(22)
#     x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
#     f = function([], T.exp(x))
#     print(f.maker.fgraph.toposort())
#     t0 = time.time()
#     for i in range(iters):
#         r = f()
#     t1 = time.time()
#     print("Looping %d times took %f seconds" % (iters, t1 - t0))
#     print("Result is %s" % (r,))
#     if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
#         print('Used the cpu')
#     else:
#         print('Used the gpu')
# Perform check:
#testTheano()

