#https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import StatFunctions


def LassoRegression(x, y, z, degree=5, alpha=10**(7), verbose=False):
    # Split into training and test
    x_train = np.random.rand(100,1)
    y_train = np.random.rand(100,1)
    z = StatFunctions.FrankeFunction(x_train,y_train)

    # train and find design matrix X_
    X = np.c_[x_train,y_train]
    poly = PolynomialFeatures(degree)
    X_ = poly.fit_transform(X)
    clf = linear_model.Lasso(alpha, fit_intercept=False)
    clf.fit(X_, z)
    beta = clf.coef_

    # predict
    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)
    M = np.c_[x, y]
    M_ = poly.fit_transform(M)
    predict = M_.dot(beta.T)

    if verbose:
        print ("x: ", np.shape(x))
        print ("y: ", np.shape(y))
        print ("M: ", np.shape(M))
        print ("M_: ", np.shape(M_))
        print ("predict: ", np.shape(predict))

    design = np.dot(np.transpose(X),X)
    # plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_,y_,predict.reshape(20,20),
    linewidth=0,
    antialiased=False)
    plt.show()

    return beta, design

if __name__ == '__main__':
    x = np.arange(0, 1, 0.05).reshape((20,1))
    y = np.arange(0, 1, 0.05).reshape((20,1))
    #z = StatFunctionsc.(x,y)
    beta, design = LassoRegression(x,y,5,alpha=10**(9),verbose=True)
   
    print ("ok")