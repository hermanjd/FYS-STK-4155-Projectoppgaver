#Ressource:
#https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#id1

#cross-validation with f-fold 
from math import degrees
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import StatFunctions

def RidgeRegression(x, y, poly, lambdas, kfold):   
    
    i = 0
    for lmb in lambdas:
        ridge = Ridge(alpha = lmb)
        j = 0
        for train_inds, test_inds in kfold.split(x):
            xtrain = x[train_inds]
            ytrain = y[train_inds]

            xtest = x[test_inds]
            ytest = y[test_inds]

            Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
            ridge.fit(Xtrain, ytrain[:, np.newaxis])

            Xtest = poly.fit_transform(xtest[:, np.newaxis])
            ypred = ridge.predict(Xtest)

            scores_KFold[i,j] = np.sum((ypred - ytest[:, np.newaxis])**2)/np.size(ypred)

            j += 1
        i += 1

    z = StatFunctions.FrankeFunction(xtrain,ytrain)
    mse = StatFunctions.evaluateMSE(z,y)
    mse_KFold = np.mean(scores_KFold, axis = 1)
    mse_sklearn = np.zeros(nlambdas)
    i = 0
    for lmb in lambdas:
        ridge = Ridge(alpha = lmb)

        X = poly.fit_transform(x[:, np.newaxis])

        #mse[i] = np.float64(mse)
        #print (mse)
        # cross_val_score
        mse_KFold = cross_val_score(ridge, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)

        # cross_val_score return an array for every fold.
        mse_sklearn[i] = np.mean(-mse_KFold)

        i += 1

    #plot
    plt.figure()
    plt.plot(np.log10(lambdas), mse_sklearn, label = 'cross_val_score')
    plt.plot(np.log10(lambdas), mse_KFold, 'r--', label = 'KFold')
    plt.plot(np.log10(lambdas), mse, 'r--', label = 'MSE')
    plt.ylabel('mse')
    plt.legend()
    plt.show()


#Driven Methode
if __name__=='__main__':
    print("Task D")
    #seed to ensure the random numbers is same to eavery run
    np.random.seed(2345)

    #Generate tha data
    nsample = 50
    x=np.random.randn(nsample)
    y=np.random.randn(nsample)

    #degree in polynomial 
    poly=PolynomialFeatures(degree = 8)

    #decied which values of lambda to use
    nlambdas=66
    lambdas=np.logspace(0, 5, nlambdas)

    #initialize a k-Fold
    k=7
    kfold = KFold(n_splits = k)

    #Perform the cross-validation to estimate MSE
    scores_KFold = np.zeros((nlambdas, k))

    #calling RidgeRegression
    object= RidgeRegression(x, y, poly, lambdas, kfold)

