import mnist_loader
import network2 as network
import numpy as np
import sys

training_data, validation_data, test_data = mnist_loader.load_data_wrapper('./mnist/data/mnist.pkl.gz')

def relu(x):
    return np.where(x > 0, x, x * 0.01)

def reluPrime(x):
    return np.where(x >= 0, 1, 0.01)

net = network.Network([784, 30,30, 10], acti=relu, actiPrime=reluPrime)

net.SGD(training_data, 30, 10, 0.5, test_data=test_data)
