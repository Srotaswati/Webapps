import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

import h5py
from scipy import ndimage
from PIL import Image #after ! pip install Pillow


def load_flower():
    np.random.seed(1)
    m = 300 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
  
    return X, Y
    
    
def load_dataset():
    N = 300
    np.random.seed(3)
    flower = load_flower()
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return flower, noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure


def get_training_data(dataset):
    flower, noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_dataset()
    datasets = {'a': flower, 'b': noisy_circles, 'c': noisy_moons,'d': blobs, 'e': gaussian_quantiles}
    train_x, train_y = datasets[dataset]
    train_x, train_y = train_x.T, train_y.reshape(1, train_y.shape[0])
    if dataset == "blobs":
        train_y = train_y%2
    return train_x, train_y
