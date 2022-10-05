import numpy as np
from matplotlib import cm
import numpy as np
import StatFunctions
from sklearn.model_selection import train_test_split

N = 100
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = StatFunctions.FrankeFunctionWithNoise(x,y,0.1) #adding some noise to the data
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x,y,z,test_size=0.2)
for i in range(0, 6):
	X = StatFunctions.create_X(x_train,y_train,i)
	B = StatFunctions.findBetaValues(X,z_train)
	testMatrix = StatFunctions.create_X(x_test,y_test,i)
	Y = StatFunctions.findY(testMatrix,B)
	mse = StatFunctions.evaluateMSE(z_test,Y)
	RSquared = StatFunctions.evaluateRSquared(z_test,Y)
	print("TEST- Polynomial degree: {} Mean square error: {} RSquared: {}".format(i,mse,RSquared))
