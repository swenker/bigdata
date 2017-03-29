__author__ = 'wenjusun'

from numpy import *
from sklearn.neighbors import *


def get_dataset():
    pass

def kNN():
    X = array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    print distances,indices

def testit():
    kNN()


if __name__ == "__main__":
    testit()