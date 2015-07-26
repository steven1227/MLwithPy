from numpy.ma import log
from scipy import special, optimize

__author__ = 'steven'
import os
from matplotlib import pyplot
import numpy as np

__author__ = 'steven'
data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + '/ex2data2.txt', delimiter=',', dtype="float")

negative = data[data[:, 2] == 0];
positive = data[data[:, 2] == 1];

pyplot.scatter(negative[:, 0], negative[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted")
pyplot.scatter(positive[:, 0], positive[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted")


def magfeatrue(x1, x2):
    m = x1.size
    degree = 6
    final = np.c_[np.ones(m)]
    mag_1 = None;
    mag_2 = None;
    for i in range(1, 7, 1):
        for j in range(i + 1):
            temp = (x1 ** j) * (x2 ** (i - j))
            final = np.c_[final, temp]
    return final


def sigmoid(z):
    return special.expit(z)


def hypo(x, theta):
    return sigmoid(x.dot(theta))


def computeCost(theta, X, y, lamda):
    m = np.shape(X)[0]
    hypo = sigmoid(X.dot(theta))
    term1 = log(hypo).dot(-y)
    term2 = log(1.0 - hypo).dot(1 - y)
    left_hand = (term1 - term2) / m
    right_hand = theta.transpose().dot(theta) * lamda / (2 * m)
    return left_hand + right_hand


def cost_func(theta, x, y, lamda):
    m = y.size
    temp1 = log(hypo(x, theta)).dot(-y)
    temp2 = log(1.0 - hypo(x, theta)).dot(1.0 - y)
    return 1.0 / m * (temp1 - temp2) + lamda * (np.sum(theta ** 2))/ (m * 2)


def findMinTheta(theta, x, y, lamda):
    result = optimize.minimize(cost_func, x0=theta, args=(x, y, lamda), method='BFGS',
                               options={"maxiter": 500, "disp": True})
    return result.x, result.fun


X = data[:, 0:2]
Y = data[:, 2:3]
Lamda = 1
X = magfeatrue(X[:, 0], X[:, 1]);
theta_initial = np.zeros(X.shape[1])

print cost_func(theta_initial, X, Y, 1)
theta_final, come = findMinTheta(theta_initial, X, Y, 1)

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))

for i in range(0, len(u)):
    for j in range(0, len(v)):
        mapped = magfeatrue(np.array([u[i]]), np.array([v[j]]))
        z[i, j] = mapped.dot(theta_final)
z = z.transpose()

u, v = np.meshgrid(u, v)
pyplot.contour(u, v, z, [0.0, 0.0], label='Decision Boundary')

pyplot.show()
