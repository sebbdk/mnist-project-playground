import numpy as np
from urllib import request
import gzip
import pickle
import os

filename = [
	["training_images","train-images-idx3-ubyte.gz"],
	["test_images","t10k-images-idx3-ubyte.gz"],

	["training_labels","train-labels-idx1-ubyte.gz"],
	["test_labels","t10k-labels-idx1-ubyte.gz"]
]

basePath = './from_scratch/mnist_np/';
tmpPath = basePath +'.tmp/';

def download_mnist():
    if not os.path.exists(tmpPath):
        os.makedirs(tmpPath)

    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], tmpPath + name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}

    # Load last 2
    for name in filename[:2]:
        with gzip.open(tmpPath + name[1], 'rb') as f:
          d = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)/255
          d.shape += (1,)

        mnist[name[0]] = d.astype(np.float32)

    # Load first 2
    for name in filename[-2:]:
        #print(name)
        d = []
        with gzip.open(tmpPath + name[1], 'rb') as f:
            d = np.frombuffer(f.read(), np.uint8, offset=8)

        mnist[name[0]] = d

    # Save to pkl
    with open( basePath + "mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()