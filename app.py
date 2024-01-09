import mnist_loader
import network2 as network
import numpy as np

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper('./mnist/data/mnist.pkl.gz')
training_data, validation_data, test_data = mnist_loader.load_data_wrapper_pickled()

print(training_data[0])

#test 4

def npRelu(x):
    return np.where(x > 0, x, 0)

def reluPrime(x):
    return np.where(x >= 0, 1, 0.01)


#dl.save_mnist()
#net = network.Network([784, 32, 10], acti=npRelu, actiPrime=reluPrime, showProgress=True)
net = network.Network([784, 32, 10], acti=npRelu, actiPrime=reluPrime)
#net = network.Network([784, 30, 10])

#net.SGD(training_data, 30, 10, 3, test_data=test_data, debug=True)
net.SGD(training_data, 500, 10, 0.01, test_data=test_data, debug=True)
