import numpy as np
from matplotlib import cm
import numpy as np
import StatFunctions
import matplotlib.pyplot as plt

N = 100
polynomialDegrees = 5
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = StatFunctions.FrankeFunctionWithNoise(x,y,0.0) #adding some noise to the data
polydegree_bootstrap, error_bootstrap, bias_bootstrap, variance_bootstrap = StatFunctions.PolynomialOLSBootstrapResampling(x,y,z,0.2,polynomialDegrees,N)
polydegree_crossValidation, error_crossValidation = StatFunctions.PolynomialOLSCrossValidation(x,y,z,polynomialDegrees,20)

plt.plot(polydegree_bootstrap, error_crossValidation, label='Crossvalidation')
plt.plot(polydegree_bootstrap, error_bootstrap, label='Bootstrap')
plt.legend()
plt.show()