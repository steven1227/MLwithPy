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


def paramUnroll(nn_params, input_layer_size, hidden_layer_size,
                num_labels):  # Actually, I do not know what are you doing...
    theta1_elems = (input_layer_size + 1) * hidden_layer_size
    theta1_size = (input_layer_size + 1, hidden_layer_size)
    theta2_size = (hidden_layer_size + 1, num_labels)

    theta1 = nn_params[:theta1_elems].T.reshape(theta1_size).T
    theta2 = nn_params[theta1_elems:].T.reshape(theta2_size).T

    return (theta1, theta2)


def sigmoid(z):
    return scipy.special.expit(z)


def recodeLabel(y, k):
    m = shape(y)[0]
    out = zeros((k, m))
    for i in range(0, m):
        out[y[i] - 1, i] = 1
    return out


def feedForward(theta1, theta2, X, X_bias):
    one_rows = ones((1, shape(X)[0]))
    a1 = r_[one_rows, X.T] if X_bias is None else X_bias
    z2 = theta1.dot(a1)
    a2 = sigmoid(z2)
    a2 = r_[one_rows, a2]
    z3 = theta2.dot(a2)
    a3 = sigmoid(z3)
    return (a1, a2, a3, z2, z3)


def computeCost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda, yk=None, X_bias=None):
    m, n = shape(X)
    theta1, theta2 = paramUnroll(nn_params, input_layer_size, hidden_layer_size, num_labels)
    a1, a2, a3, z2, z3 = feedForward(theta1, theta2, X, X_bias)

    if yk is None:
        yk = recodeLabel(y, num_labels)
        assert shape(yk) == shape(a3), "Error, shape of recoded y is different from a3"
    print theta1.shape, theta2.shape
    term1 		= -yk * log( a3 )
    term2 		= (1 - yk) * log( 1 - a3 )
    left_term 	= sum(term1 - term2) / m
    right_term 	= sum(theta1[:,1:] ** 2) + sum(theta2[:,1:] ** 2)
    return left_term + right_term * lamda / (2 * m)


mat = scipy.io.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/ex4data1.mat')
mat2 = scipy.io.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/ex4weights.mat')
X, Y = mat['X'], mat['y']
theta1, theta2 = mat2['Theta1'], mat2['Theta2']
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lamda = 1
displayData(X)

# a = array([[10, 20], [30, 40]])
# b = array([[1, 2], [3, 4]])
# print r_[a.flatten(), b.flatten()]
params = r_[theta1.T.flatten(), theta2.T.flatten()]
print theta1.shape, theta2.shape
# print X.shape
print params.shape
print computeCost(params, input_layer_size, hidden_layer_size, num_labels, X, Y, lamda)
