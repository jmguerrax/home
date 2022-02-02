import time
import numpy as np
import matplotlib.pyplot as plt

### Create training data: 

N_tot = 200 #Total samples

x_min = -1 # x-domain
x_max = 1

A_c = 1 
x = np.linspace(x_min, x_max, N_tot) 

def f_t(x):
    #return A_c*np.cos(x)
    return A_c*np.power(x,2)

plt.plot(x,f_t(x),label='Original data')
plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.xlabel('x')
plt.ylabel('f_t(x)')
plt.grid()
plt.show()

