import numpy as np
from matplotlib import cm
import numpy as np
from FrankeFunction import FrankeFunctionWithNoise, FrankeFunction
from MakeDesignMatrix import makeDesignMatrix

def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def findBetaValues(X,y):
	XT = X.transpose()
	core = np.matmul(XT, X)
	coreInversed = np.linalg.inv(core)
	XTY = np.matmul(coreInversed, XT)
	B = np.matmul(XTY,y)
	return B

def findY(X,b):
	values = []
	for index_row, element_row in enumerate(X):
		row = 0.0;
		for index_column, element_column in enumerate(element_row):
			row = row + (element_column*b[index_column])
		values.append(row)
	return values

def evaluateMean(Y):
	sum = 0.0
	length = len(Y)
	for i in range(length):
		sum = sum + Y[i]
	return sum * (1/length)

def evaluateMSE(Y,y):
	sum = 0.0
	length = len(Y)
	for i in range(length):
		sum = sum + pow((Y[i]-y[i]),2)
	return sum * (1/length)

def evaluateRSquared(Y,y):
	MSE = evaluateMSE(Y,y)
	mean = evaluateMean(Y)
	sum = 0.0
	length = len(Y)
	for i in range(length):
		sum = sum + pow((Y[i]-mean),2)
	return 1 - (MSE/sum)

N = 1000000
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = FrankeFunction(x,y)
X = create_X(x,y,5)

B = findBetaValues(X,z)
Y = findY(X,B)
mse = evaluateRSquared(z,Y)
print(mse)