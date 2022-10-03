import numpy as np
from matplotlib import cm
import numpy as np
import StatFunctions

N = 1000000
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = StatFunctions.FrankeFunction(x,y)
X = StatFunctions.create_X(x,y,5)
B = StatFunctions.findBetaValues(X,z)
Y = StatFunctions.findY(X,B)
mse = StatFunctions.evaluateRSquared(z,Y)
print(mse)