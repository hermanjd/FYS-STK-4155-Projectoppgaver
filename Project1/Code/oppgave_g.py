import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import StatFunctions
from sklearn.model_selection import train_test_split
# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')

nparray = np.array(terrain1)[:-1, :-1]

y_raw, x_raw = nparray.shape

print(y_raw)
x = np.arange(0,x_raw)[0::2]
y = np.arange(0,y_raw)[0::2]

xs = np.arange(0,x_raw)
ys = np.arange(0,y_raw)
xvs, yvs = np.meshgrid(xs, ys)

xv, yv = np.meshgrid(x, y)
x_ready = xv.flatten()
y_ready = yv.flatten()
z_ready = []
for i, yv in enumerate(y_ready):
    z_ready.append(nparray[yv][x_ready[i]])

tonp = np.array(z_ready)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_ready,y_ready,tonp,test_size=0.2)

for i in range(27, 28):
    print("starting experiment")
    print(i)
    X = StatFunctions.create_X(x_train,y_train,i)
    B = StatFunctions.findBetaValues(X,z_train)
    Xpredictor = StatFunctions.create_X(x_test,y_test,i)
    Y = StatFunctions.findY(Xpredictor,B)
    mse = np.mean((z_test-Y)**2)
    print(mse)


