import scipy.io
import os
import numpy as np
from matplotlib import pyplot

__author__ = 'steven'

def dist(x1, x2):
    cost = (x1-x2)
    return cost.T.dot(cost)


def findClosetCentroid(x, centroid):
    idx= np.zeros((x.shape[0],1))
    for i in range(idx.size):
        cost=[]
        for j in centroid:
            cost.append(dist(x[i], j))
        idx[i] = cost.index(min(cost))+1
    return idx

def ComputeCentroids(x, idx, K):
    center_new= np.zeros((K, x.shape[1]))
    for i in range(K):
        temp = x[(i+1) == idx[:,0]]
        center_new[i] = [np.sum(temp[:,0])/temp.shape[0],np.sum(temp[:,1])/temp.shape[0]]
    return center_new

mat = scipy.io.loadmat(os.getcwd() + "/ex7data2.mat")
X = mat['X']
K = 3

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = findClosetCentroid(X, initial_centroids)
centroids = ComputeCentroids(X, idx, K)
pyplot.plot(X[:, 0], X[:, 1], 'bx', markersize=5)
pyplot.show()
