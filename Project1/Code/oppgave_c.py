import numpy as np
from matplotlib import cm
import numpy as np
import StatFunctions
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt

N = 50
polynomialDegrees = 5
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = StatFunctions.FrankeFunctionWithNoise(x,y,0.0) #adding some noise to the data
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x,y,z,test_size=0.2)

error = np.zeros(polynomialDegrees+1)
bias = np.zeros(polynomialDegrees+1)
variance = np.zeros(polynomialDegrees+1)
polydegree = np.zeros(polynomialDegrees+1)

for i in range(0, polynomialDegrees+1):
    print("-----------Polynomial degree: {}-------------------".format(i))
    z_pred = []
    
    for j in range(N):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        designMatrix = StatFunctions.create_X(x_,y_,i)
        B = StatFunctions.findBetaValues(designMatrix,z_)
        testMatrix = StatFunctions.create_X(x_test,y_test,i)
        Y = StatFunctions.findY(testMatrix,B)
        z_pred.append(np.asarray(Y))
    
    arr = np.asarray(z_test)
    arr2 = np.asarray(z_pred)
    temp_error = np.mean( np.mean((arr - arr2)**2, axis=0, keepdims=True) )
    temp_bias = np.mean( (arr - np.mean(arr2, axis=0, keepdims=True))**2 )
    temp_variance = np.mean( np.var(arr2, axis=0, keepdims=True) )
    print('Error:', temp_error)
    print('Bias^2:', temp_bias)
    print('Var:', temp_variance)
    print('{} >= {} + {} = {}'.format(temp_error, temp_bias, temp_variance, temp_bias+temp_variance))

    polydegree[i] = i
    error[i] = temp_error
    bias[i] = temp_bias
    variance[i] = temp_variance
    #print(np.mean( np.mean((arr - z_pred)**2, axis=1, keepdims=True) ))
    


plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()
