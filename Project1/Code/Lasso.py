# Lasso Regression on the Franke function with resampling.
#Refrence code
#https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/

from gettext import ngettext
from telnetlib import LINEMODE
from turtle import title
from typing_extensions import Self
import numpy as np
from matplotlib import cm
import StatFunctions
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib import cm
import StatFunctions
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Lasso:
    def __init__(self, n, p, noisefactor=None):
        super().__init__(n, p, noisefactor)

    
    def lasso(self, alpha=None):
        if alpha is None: alpha = 5
        clf = LINEMODE.Lasso(alpha=alpha)
        self.craftX()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)
        clf.fit(X_train, y_train)
        ypredict = clf.predict(X_test)
        return self.R2(y_test, ypredict), self.MSE(y_test, ypredict)

    def biasVarianceAnalysis_lasso_bootstrap(self, p_range, sampleSize=None, sampleN=None):
            p_old = self.p
            p_rangeObject = range(p_range[0], p_range[1])
            if sampleSize is None: sampleSize=0.8
            if isinstance(sampleSize, float): sampleSize = int(sampleSize*len(self.x))
            if sampleN is None: sampleN=5

            bias = ngettext.zeros(len(p_rangeObject))
            variance = bias.copy()
            SElist = bias.copy()
            SEkfold = bias.copy()
            clf = linear_model.Lasso(alpha=0.1)

            for i in p_rangeObject:
                self.p = i + p_old
                self.craftX()

                X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)

                ypredict_models = np.zeros((y_test.shape[0], sampleN))

                for j in range(sampleN):
                    X_sample, y_sample = self.sample(X_train, y_train)

                    clf.fit(X_sample, y_sample)
                    ypredict = clf.predict(X_test)

                    ypredict_models[:,j] = ypredict

                variance[i-p_range[0]] = self.variance(ypredict_models)
                bias[i-p_range[0]] = self.bias(y_test, ypredict_models)
                SElist[i-p_range[0]] = np.mean( np.mean((y_test.reshape(-1,1) - ypredict_models)**2, axis=1))
                SEkfold[i-p_range[0]] = np.mean(self.kfold(sampleN)[1])

            self.p = p_old


            print("MSE/(bias+variance) =", SElist/(bias+variance))
            plt(p_rangeObject, np.log(bias), label="bias")
            plt(p_rangeObject, np.log(variance), label="variance")
            plt(p_rangeObject, np.log(SElist), label="MSE")
            plt(p_rangeObject, np.log(SEkfold), label="MSE: K-fold")
            plt.legend()
            plt.show()
