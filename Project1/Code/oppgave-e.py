#Ridge Regression on the Franke function
import numpy as np
from matplotlib import cm
import StatFunctions
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures



def Ridge(x, y, z, polynomialDegrees, l=0.1):
    # Calculate matrix with x, y - polynomials
    M_ = np.c_[x, y]
    poly = PolynomialFeatures(polynomialDegrees)
    M = poly.fit_transform(M_)

    # Calculate beta
    A = np.arange(1, polynomialDegrees + 2)
    rows = np.sum(A)
    beta = (np.linalg.inv(M.T.dot(M) + l * np.identity(rows))).dot(M.T).dot(z)

    return beta

if __name__ == '__main__':
        N = 100
        polynomialDegrees = 5
        x = np.random.uniform(0, 1, N)
        y = np.random.uniform(0, 1, N)
        z = StatFunctions.FrankeFunctionWithNoise(x,y,0.0) #adding some noise to the data
        polydegree_bootstrap, error_bootstrap, bias_bootstrap, variance_bootstrap = StatFunctions.PolynomialOLSBootstrapResampling(x,y,z,0.2,polynomialDegrees,N)
        polydegree_crossValidation, error_crossValidation = StatFunctions.PolynomialOLSCrossValidation(x,y,z,polynomialDegrees,20)
        Ridge= Ridge(x, y, z, polynomialDegrees, N)
        plt.plot(polydegree_bootstrap, error_crossValidation, label='Crossvalidation')
        plt.plot(polydegree_bootstrap, error_bootstrap, label='Bootstrap')
        plt.plot(Ridge, label='LassoR egression')        
        plt.legend()
        plt.show()

