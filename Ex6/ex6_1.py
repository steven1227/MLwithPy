import os
from scipy import io
from sklearn import svm, grid_search
from numpy import *
from matplotlib import pyplot

__author__ = 'steven'
mat = io.loadmat(os.path.dirname(os.path.realpath(__file__)) + "/ex6data1.mat")
X, Y = mat['X'], mat['y']
mat_a = c_[X, Y]
mat_b = mat_a[mat_a[:, 2] == 1]
mat_c = mat_a[mat_a[:, 2] == 0]

pyplot.subplot(221)
pyplot.plot(mat_b[:, 0], mat_b[:, 1], 'b+')
pyplot.plot(mat_c[:, 0], mat_c[:, 1], 'yo')


def dataset3ParamsVer3(X, y, X_val, y_val):
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    gammas = map(lambda x: 1.0 / x, sigma_values)
    print gammas
    # print gammas
    raveled_y = y.ravel()

    rbf_svm = svm.SVC()
    parameters = {'kernel': ('rbf',), 'C': C_values,
                  'gamma': gammas}
    grid = grid_search.GridSearchCV(rbf_svm, parameters)
    best = grid.fit(X, raveled_y).best_params_

    return best


def dataset3ParamsVer1(X, y, X_val, y_val):
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    raveled_y = y.ravel()  # Else the SVM will give you annoying warning
    m_val = shape(X_val)[0]  # number of entries in validation data

    rbf_svm = svm.SVC(kernel='rbf')

    best = {'error': 999, 'C': 0.0, 'sigma': 0.0}

    for C in C_values:
        for sigma in sigma_values:
            # train the SVM first
            rbf_svm.set_params(C=C)
            rbf_svm.set_params(gamma=1.0 / sigma)
            rbf_svm.fit(X, raveled_y)

            # test it out on validation data
            predictions = []
            for i in range(0, m_val):
                prediction_result = rbf_svm.predict(X_val[i])
                predictions.append(prediction_result[0])

            # sadly if you don't reshape it, numpy doesn't know if it's row or column vector
            predictions = array(predictions).reshape(m_val, 1)
            error = (predictions != y_val.reshape(m_val, 1)).mean()

            # get the lowest error
            if error < best['error']:
                best['error'] = error
                best['C'] = C
                best['sigma'] = sigma

    best['gamma'] = 1.0 / best['sigma']
    return best


def visualizeBoundary(X, trained_svm):
    kernel = trained_svm.get_params()['kernel']
    if kernel == 'linear':
        w = trained_svm.dual_coef_.dot(trained_svm.support_vectors_).flatten()
        xp = linspace(min(X[:, 0]), max(X[:, 0]), 100)
        # print trained_svm.intercept_
        yp = (-w[0] * xp - trained_svm.intercept_) / w[1]
        pyplot.plot(xp, yp, 'b')

    elif kernel == 'rbf':
        x1plot = linspace(min(X[:, 0]), max(X[:, 0]), 100)
        x2plot = linspace(min(X[:, 1]), max(X[:, 1]), 100)
        X1, X2 = meshgrid(x1plot, x2plot)
        vals = zeros(shape(X1))

        for i in range(0, shape(X1)[1]):
            this_X = c_[X1[:, i], X2[:, i]]
            vals[:, i] = trained_svm.predict(this_X)

        pyplot.contour(X1, X2, vals, colors='blue')


linear_svm = svm.SVC(C=100, kernel='linear')
linear_svm.fit(X, Y.ravel())
# linear_svm.predict([[2.5, 4.]])
# linear_svm.dual_coef_.dot(linear_svm.support_vectors_).flatten()
visualizeBoundary(X, linear_svm)


# Compute the Gaussian kernel
x1 = array([1, 2, 1])
x2 = array([0, 4, -1])
sigma = 2


def gaussianKernel(x1, x2, sigma):
    return exp(-((x1 - x2).dot((x1 - x2).T)) / 2.0 / sigma ** 2)


print gaussianKernel(x1, x2, sigma)

mat = io.loadmat(os.path.dirname(os.path.realpath(__file__)) + "/ex6data2.mat")
X, Y = mat['X'], mat['y']
mat_a = c_[X, Y]
mat_b = mat_a[mat_a[:, 2] == 1]
mat_c = mat_a[mat_a[:, 2] == 0]
pyplot.subplot(222)
pyplot.plot(mat_b[:, 0], mat_b[:, 1], 'b+')
pyplot.plot(mat_c[:, 0], mat_c[:, 1], 'yo')

sigma = 0.01
rbf_svm = svm.SVC(C=1, kernel='rbf', gamma=1.0 / sigma)  # gamma is actually inverse of sigma
rbf_svm.fit(X, Y.ravel())
visualizeBoundary(X, rbf_svm)

mat = io.loadmat(os.path.dirname(os.path.realpath(__file__)) + "/ex6data3.mat")
X, Y = mat['X'], mat['y']
mat_a = c_[X, Y]
mat_b = mat_a[mat_a[:, 2] == 1]
mat_c = mat_a[mat_a[:, 2] == 0]
X_val, y_val = mat['Xval'], mat['yval']
pyplot.subplot(223)
pyplot.plot(mat_b[:, 0], mat_b[:, 1], 'b+')
pyplot.plot(mat_c[:, 0], mat_c[:, 1], 'yo')

rbf_svm = svm.SVC(kernel='rbf')

best = dataset3ParamsVer3(X, Y, X_val, y_val)
rbf_svm.set_params(C=best['C'])
rbf_svm.set_params(gamma=best['gamma'])
rbf_svm.fit(X, Y.ravel())
visualizeBoundary(X, rbf_svm)
pyplot.show(block=True)
