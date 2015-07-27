import os
from scipy import special

__author__ = 'steven'

import PIL.Image
import scipy.misc, scipy.optimize, scipy.io
from numpy import *

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab


def sigmoid(z):
    return special.expit(z)


def cost_func(theta, x, y, lamda):
    m = y.size
    temp1 = log(hypo(x, theta)).dot(-y)
    temp2 = log(1.0 - hypo(x, theta)).dot(1.0 - y)
    return 1.0 / m * (temp1 - temp2) + lamda * (sum(theta ** 2)) / (m * 2)


def hypo(x, theta):
    return sigmoid(x.dot(theta))


def gradientCost(theta, x, y, lamda):
    m = y.size
    cost = x.T.dot((hypo(x, theta) - y))/m
    cost[1:] = cost[1:] + ((theta[1:] * lamda) / m)
    return cost / m


def displayData(X, theta=None):
    width = 20
    rows, cols = 10, 10
    out = zeros((width * rows, width * cols))
    rand_indices = random.permutation(5000)[0:rows * cols]
    counter = 0
    for y in range(0, rows):
        for x in range(0, cols):
            start_x = x * width
            start_y = y * width
            out[start_x:start_x + width, start_y:start_y + width] = X[rand_indices[counter]].reshape(width, width).T
            counter += 1
    img = scipy.misc.toimage(out)
    figure = pyplot.figure()
    axes = figure.add_subplot(111)
    axes.imshow(img)

    if theta is not None:
        result_matrix = []
        X_biased = c_[ones(shape(X)[0]), X]

        for idx in rand_indices:
            result = (argmax(theta.T.dot(X_biased[idx])) + 1) % 10
            result_matrix.append(result)

        result_matrix = array(result_matrix).reshape(rows, cols).transpose()
        print result_matrix

    pyplot.show()


def oneVsAll(X, y, num_classes, lamda):
    m, n = shape(X)
    X = c_[ones((m, 1)), X]
    all_theta = zeros((n + 1, num_classes))

    for k in range(0, num_classes):
        theta = zeros((n + 1, 1)).reshape(-1)
        temp_y = ((y == (k + 1)) + 0).reshape(-1)
        result = scipy.optimize.fmin_cg(cost_func, fprime=gradientCost, x0=theta, args=(X, temp_y, lamda), maxiter=50, disp=False, full_output=True)
        all_theta[:, k] = result[0]
        print "%d Cost: %.5f" % (k + 1, result[1])

    # save( "all_theta.txt", all_theta )
    return all_theta


mat = scipy.io.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/ex3data1.mat')
X, Y = mat['X'], mat['y']

m = X.shape[0]
# X = c_[ones((m, 1)), X]

theta_initial = zeros((X.shape[1], 1))

theta = oneVsAll( X, Y, 10, 0.1)
# displayData(X, theta_initial)
# gradientCost(theta_initial, X, Y, 1)
displayData(X, theta)