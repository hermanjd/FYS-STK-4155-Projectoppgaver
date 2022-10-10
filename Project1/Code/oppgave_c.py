import numpy as np
from matplotlib import cm
import numpy as np
import StatFunctions
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt

N = 65
polynomialDegrees = 5
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = StatFunctions.FrankeFunctionWithNoise(x,y,0.0) #adding some noise to the data
polydegree, error, bias, variance =StatFunctions.PolynomialOLSBootstrapRegression(x,y,z,0.2,polynomialDegrees,N)
plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()
