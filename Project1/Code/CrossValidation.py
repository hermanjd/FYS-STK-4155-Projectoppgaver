import numpy as np
from matplotlib import cm
import numpy as np
import StatFunctions
import matplotlib.pyplot as plt

N = 60
polynomialDegrees = 5
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = StatFunctions.FrankeFunctionWithNoise(x,y,0.0) #adding some noise to the data
polydegree, error, bias, variance = StatFunctions.PolynomialOLSCrossValidation(x,y,z,polynomialDegrees,5)
plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()
