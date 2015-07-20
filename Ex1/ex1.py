import os
import sys
import numpy as np
import scipy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

file_1 = open(os.path.dirname(os.path.realpath(__file__)) + '/ex1data1.txt', 'r')
table = [row.strip().split(',') for row in file_1]

array = np.asarray(table, dtype=np.float_)
population = array[:, 0]
profit = array[:, 1]
pyplot.plot(population, profit, 'x', color='r')
pyplot.axis([4, 24, -5, 25])
pyplot.xlabel("Population")
pyplot.ylabel("Profit")

m = len(profit)
iterations = 1500
alpha = 0.01
theta = np.zeros((2, 1))
# theta[1,0]=1
y = profit.reshape(m, 1)
x = np.c_[np.ones(m), population]


def computecost(x, y, theta):
    cost = 1.0 / (2 * m) * np.sum((hypothesis(x, theta) - y) ** 2)
    return cost


def hypothesis(x, theta):
    return np.dot(x, theta)


def graient(x, y, theta_init, alpha, iterations):
    grad = np.copy(theta_init)
    for i in range(0, iterations):
        temp_sum = (hypothesis(x, grad) - y) * x
        grad = grad - alpha / m * np.array([[np.sum(temp_sum[:, 0])], [np.sum(temp_sum[:, 1])]])
    return grad


grad = graient(x, y, theta, alpha, iterations)

pyplot.plot(population, hypothesis(x, grad), '-x')
pyplot.show()
