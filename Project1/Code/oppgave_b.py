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
			print("x^" + str(i-k) + "*y^" + str(k))
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def findBetaValues(X,y):
	XT = X.transpose()
	core = np.matmul(XT, X)
	coreInversed = np.linalg.inv(core)
	XTY = np.matmul(coreInversed, XT)
	B = np.matmul(XTY,y)
	return B

N = 10000000
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = FrankeFunction(x,y)
X = create_X(x,y,5)

B = findBetaValues(X,z)
print(B)

