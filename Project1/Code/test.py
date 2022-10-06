from collections import namedtuple
from re import X

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import FF
from FF import *
from PolynomialFittingData import *
import Lasso


# Create Data
Array1 = [4, 30, 84, 5]
Array2 = [7, 75, -3, 5]

# Driver Function
#Polynomial Fitting Data Test with 1D
#object = PolynomialFittingData(Array1, Array2)

#Frank Polynomial Fitting Data Test with 2D
#object = FF(Array1, Array2)

#laso
N = 50
polynomialDegrees = 5
polynomialDegrees_Range = [0, 5]
alphas=[0.4, 0.8, 0.9, 0.15]
sampleN = 10
error = np.zeros(polynomialDegrees+1)
bias = np.zeros(polynomialDegrees+1)
variance = np.zeros(polynomialDegrees+1)
polydegree = np.zeros(polynomialDegrees+1)



