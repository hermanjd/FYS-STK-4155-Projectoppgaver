import numpy as np
from matplotlib import cm
import numpy as np
import StatFunctions
from sklearn.model_selection import train_test_split

N = 4000
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = StatFunctions.FrankeFunctionWithNoise(x,y,0.02)

for i in range(0, 6):
	X = StatFunctions.create_X(x,y,i)
	X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
	B = StatFunctions.findBetaValues(X_train,z_train)
	Y = StatFunctions.findY(X_test,B)
	mse = StatFunctions.evaluateMSE(z_test,Y)
	RSquared = StatFunctions.evaluateRSquared(z_test,Y)
	print("Plynomial degree: {} Mean square error: {} RSquared: {}".format(i,mse,RSquared))
