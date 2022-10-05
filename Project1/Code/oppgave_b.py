import numpy as np
from matplotlib import cm
import numpy as np
import StatFunctions
from sklearn.model_selection import train_test_split

N = 50
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = StatFunctions.FrankeFunctionWithNoise(x,y,0.1) #adding some noise to the data

for i in range(0, 6):
	X = StatFunctions.create_X(x,y,i)
	X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
	B = StatFunctions.findBetaValues(X_train,z_train)
	YTrain = StatFunctions.findY(X_train,B)
	Y = StatFunctions.findY(X_test,B)
	mseTrain = StatFunctions.evaluateMSE(z_train,YTrain)
	mse = StatFunctions.evaluateMSE(z_test,Y)
	RSquaredTrain = StatFunctions.evaluateRSquared(z_train,YTrain)
	RSquared = StatFunctions.evaluateRSquared(z_test,Y)
	print("TRAIN Polynomial degree: {} Mean square error: {} RSquared: {}".format(i,mseTrain,RSquaredTrain))
	print("TEST- Polynomial degree: {} Mean square error: {} RSquared: {}".format(i,mse,RSquared))
