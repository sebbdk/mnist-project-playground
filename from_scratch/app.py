import mnist_loader
import network2 as network
import numpy as np
import pathlib

dataPath = str(pathlib.Path(__file__).parent) + '/mnist/data/mnist.pkl.gz';

training_data, validation_data, test_data = mnist_loader.load_data_wrapper(dataPath)
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper_pickled()

def npRelu(x):
    return np.where(x > 0, x, 0)

def reluPrime(x):
    return np.where(x >= 0, 1, 0.01)

# med ReLu
net = network.Network([784, 32, 10], acti=npRelu, actiPrime=reluPrime)

# med Sigmoid (den defaulter til det)
#net = network.Network([784, 30, 10])

net.SGD(training_data, 500, 10, 0.01, test_data=test_data, debug=True)
