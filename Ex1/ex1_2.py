import os
import numpy as np
import scipy
from matplotlib import pyplot as py

data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + '/ex1data2.txt', delimiter=',', dtype="int")
X = data[:, 0:2]
Y = data[:, 2:3]
X1 = data[:, 0]
X2 = data[:, 1]
X1 = X1.reshape(X1.size, 1)
X2 = X2.reshape(X2.size, 1)
m = X1.__len__()
mean_1 = X1.mean()
mean_2 = X2.mean()




def hypothesis(x, theta):
    return np.dot(x, theta)


def featureNormal(raw):
    mean = raw.mean(axis=0)
    data_norm = raw - mean
    sigma = np.std(data_norm, axis=0, ddof=1)
    data_norm = data_norm / sigma
    return data_norm, mean, sigma

def computecost(x, y, theta):
    cost = 1.0 / (2 * m) * np.sum((hypothesis(x, theta) - y) ** 2)
    return cost

def gradient(x,y,theta,iteration,alpha):
    cost_temp=[]
    grad=np.copy(theta)
    for i in range(iteration):
        temp=(hypothesis(x, grad) - y)
        grad-=alpha/m*(np.dot(x.T,temp))
        cost_temp.append(computecost(x,y,grad))
    return grad,cost_temp
iterations = 1500;
alpha = 0.05;
data_after, mean, sigma = featureNormal(X)
feature_no = data_after.shape[1]
data_after = np.c_[np.ones(m), data_after]
theta_initial=np.array(np.zeros((feature_no+1,1)))

grad_final,cost_final=gradient(data_after,Y,theta_initial,iterations,alpha)
py.plot(cost_final)
print cost_final[1]
py.axis([0.0, 50.0, 0, 70000000000])
py.show()



