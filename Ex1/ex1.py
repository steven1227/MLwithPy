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
pyplot.subplot(211)
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
    cost_temp=[]
    grad = np.copy(theta_init)
    for i in range(0, iterations):
        temp_sum = (hypothesis(x, grad) - y) * x
        grad = grad - alpha / m * np.array([[np.sum(temp_sum[:, 0])], [np.sum(temp_sum[:, 1])]])
        cost_temp.append(computecost(x,y,grad))
    return grad,cost_temp


grad,cost_final= graient(x, y, theta, alpha, iterations)
pyplot.plot(population, hypothesis(x, grad), '-x')
pyplot.subplot(212)
pyplot.plot(cost_final)

def test():
    atest=10
    return atest

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-4, 4, 100)
J_vals=np.zeros((100,100))
for i in range(100):
    for j in range(100):
        t=np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[i,j]=computecost(x,y,t)

R, P = np.meshgrid(theta0_vals, theta1_vals)
fig=pyplot.figure(2)
ax=fig.gca(projection='3d')
ax.plot_surface(R,P,J_vals)
fig = pyplot.figure()
pyplot.contourf(R,P,J_vals.T,np.logspace(-2,3,20))
pyplot.show(block=True)

print
