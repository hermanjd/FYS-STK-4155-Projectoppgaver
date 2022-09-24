#Polynomial Fitting For 1D.
from re import X
import numpy as np
import matplotlib.pyplot as plt

class PolynomialFittingData:
    ##Polynomial Fitting For 1D.
    def __init__(self, Array1, Array2):
      self.Array1 = Array1
      self.Array1 = Array2
      #Print a message
      print('Method has been called.') 
      print("Polynomial Fitting For 1D Test Started With:", X)
      x = np.array(Array1)
      f = 1/4

      sine = np.sin(2*np.pi*f*x) + np.random.normal(scale=0.1, size=len(x))
      plt.plot(sine)
      poly = np.polyfit(x, sine, deg=5)

      fig, ax = plt.subplots()
      ax.plot(sine, label='data')
      ax.plot(np.polyval(poly, x), label='fit')
      ax.legend()
      plt.show()      
      

     ##Polynomial Fitting For 1D.
    def PF1D(Array1):
        print("Polynomial Fitting For 1D Test Started With:", X)
        x = np.array(Array1)
        f = 1/4

        sine = np.sin(2*np.pi*f*x) + np.random.normal(scale=0.1, size=len(x))
        plt.plot(sine)

        poly = np.polyfit(x, sine, deg=5)

        fig, ax = plt.subplots()
        ax.plot(sine, label='data')
        ax.plot(np.polyval(poly, x), label='fit')
        ax.legend()
        plt.show()      
    #
    

    

    
        

    
    

    



    

