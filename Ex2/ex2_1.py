import os
from numpy.ma import log

from scipy import special, optimize

import numpy as np
from matplotlib import pyplot

data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + '/ex2data1.txt', delimiter=',', dtype="float")
positive = np.array([i for i in data if i[2] == 1])
negative = data[data[:, 2] == 0]
X = data[:, 0:2]
Y = data[:, 2:3]

pyplot.scatter(negative[:, 0], negative[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted")
pyplot.scatter(positive[:, 0], positive[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted")


def sigmoid(z):
    return special.expit(z)


def hypothesis(x, theta):
    return sigmoid(np.dot(x, theta))


def findMinTheta(theta, x, y):
    result = optimize.fmin(computeCost, x0=theta, args=(x, y), maxiter=400, full_output=True)
    return result[0], result[1]


def computecost(theta, X, Y):
    cost_func = -Y * log(hypothesis(X, theta)) - (1 - Y) * log(1 - hypothesis(X, theta))
    cost = (np.sum(cost_func) / m).flatten()
    return cost


def computeCost(theta, X, y):
    m = np.shape(X)[0]
    hypo = sigmoid(X.dot(theta))
    # print hypo
    term1 = np.log(hypo).T.dot(-y)
    term2 = np.log(1.0 - hypo).T.dot(1 - y)
    return ((term1 - term2) / m).flatten()


m = X[:, 0].size
feature_num = X.shape[1]
theta_initial = np.zeros((feature_num + 1, 1))
X = np.c_[np.ones((m, 1)), X]

# print X
# theta_initial = np.array([[-20.5], [-0.10], [0.05]])
print theta_initial
print computeCost(theta_initial, X, Y)
print computecost(theta_initial, X, Y)
t1, t2 = findMinTheta(theta_initial, X, Y)
print t1
#
# test = np.array([1, 45, 85])
#
#
# # print hypothesis(test, t1)
# plt_x = np.arange(30, 110)
# plt_y = (t1[0] + t1[1] * plt_x) / -t1[2]
# pyplot.plot(plt_x, plt_y, color='r')
# pyplot.show()
