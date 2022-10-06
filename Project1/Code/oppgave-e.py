
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import misc
import StatFunctions


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def FindBeta(x, y, degree, alpha):
    X = np.c_[x,y]
    poly = PolynomialFeatures(degree)
    X_ = poly.fit_transform(X)
    ridge = linear_model.RidgeCV(alphas=np.array([alpha]))
    ridge.fit(X_, z)
    return ridge.coef_

def RidgeRegression(x, y, z, degree=5, alpha=10**(-6), verbose=False):
    # Split data
    x_train = np.random.rand(100,1)
    y_train = np.random.rand(100,1)
    z = FrankeFunction(x_train,y_train)
    beta = FindBeta(x, y, degree, alpha)

    # show pilot
    poly = PolynomialFeatures(degree)
    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)
    M = np.c_[x, y]
    M_ = poly.fit_transform(M)
    predict = M_.dot(beta.T)

    if verbose:
        print ("X_: ", np.shape(alpha))
        print ("M: ", np.shape(M))
        print ("M_: ", np.shape(M_))
        print ("predict: ", np.shape(predict))
    
    # show pilot
    poly = PolynomialFeatures(degree)
    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)
    M = np.c_[x, y]
    M_ = poly.fit_transform(M)
    predict = M_.dot(beta.T)
    return beta


if __name__ == '__main__':
    x = np.arange(0, 1, 0.05).reshape((20,1))
    y = np.arange(0, 1, 0.05).reshape((20,1))

    z = FrankeFunction(x,y)
    beta = RidgeRegression(x,y,z,5,5**(-3),True)
    mse = StatFunctions.evaluateMSE(x,y)
    print(mse)
    print (beta)

    beta_list = beta.ravel().tolist()
    print ("ok")

    

