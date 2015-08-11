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

def runkMeans(X, initial_centroids, max_iters, plot=False ):
    for iteration in range( 0, max_iters ):
        idx =findClosetCentroid(X,initial_centroids)
        initial_centroids = ComputeCentroids(X, idx, K)
        pyplot.plot(initial_centroids[:, 0],initial_centroids[:, 1],'gs', markersize=6)

    if plot == True:
        temp1 = X[1 == idx[:,0]]
        temp2 = X[2 == idx[:,0]]
        temp3 = X[3 == idx[:,0]]
        print temp1.shape, temp2.shape, temp3.shape

        pyplot.plot(temp1[:, 0], temp1[:, 1], 'bo', markersize=5)
        pyplot.plot(temp2[:, 0], temp2[:, 1], 'ro', markersize=5)
        pyplot.plot(temp3[:, 0], temp3[:, 1], 'yo', markersize=5)
    pass


mat = scipy.io.loadmat(os.getcwd() + "/ex7data2.mat")
X = mat['X']
K = 3

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = findClosetCentroid(X, initial_centroids)
centroids = ComputeCentroids(X, idx, K)

runkMeans(X, initial_centroids, 10, plot=True)
pyplot.show()
