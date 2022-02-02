### Kernel Ridge Regression - ML fit for a general function using python. 
### By JG

import time
import numpy as np
import matplotlib.pyplot as plt

### Create training data: 

N_tot = 200 # Total number of sample points

x_min = 0 
x_max = np.pi #1


x = np.linspace(x_min, x_max, N_tot) #np.linspace(-np.pi, np.pi, N_tot)

# Function to fit
A_c = 1
def f_t(x):
    return A_c*np.cos(x)
    #return A_c*np.power(x,3)

#plt.plot(x,f_t(x),label='Original data')
#plt.legend(loc='upper right', shadow=True, fontsize='x-large')
#plt.xlabel('x')
#plt.ylabel('f_t(x)')
#plt.grid()
#plt.show()

# Training data = % of data to train (100%-% to test/evaluate)
tr_index = 0.7

# Binary probability distribution to rearrange the data
indx = np.random.binomial(1, tr_index, len(x))

x_train, x_test = [], []
for i in range(len(x)):
    if indx[i] == 1:
        x_train.append(x[i])
    else:
        x_test.append(x[i])
 
print("Train set contains {} ({}%) elements".format(len(x_train),len(x_train)/(len(x_train)+len(x_test))*100))
print("Test set contains {} ({}%)) elements".format(len(x_test),len(x_test)/(len(x_train)+len(x_test))*100))

#plt.plot(x_train,f_t(x_train),'.',color='blue',label='Training data {}%'.format(len(x_train)/(len(x_train)+len(x_test))*100))
#plt.plot(x_test,f_t(x_test),'o',color='green',label='Test data {}%'.format(len(x_test),len(x_test)/(len(x_train)+len(x_test))*100))
#plt.legend(loc='lower left', shadow=True, fontsize='x-large')
#plt.xlabel('x')
#plt.ylabel('f(x)')
#plt.grid()
#plt.show()

### (Gaussian) Kernel definition: So different Kernels can be tested
def KER_f(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma*sigma))

### ML function:
def f_ML(x,sigma,lamb):
    sum_ML=0
    for i in range(len(x_train)):
        sum_K=0
        for j in range(len(x_train)):
            sum_K += KER_f(x_train[i],x_train[j],sigma)
        sum_ML += (f_t(x_train[i])*KER_f(x_train[i],x,sigma))/(sum_K+lamb)
    return sum_ML

Lambda = 1.0e-12 # Try
start = time.time()
print(f_ML(x[0],0.1,Lambda))
print("This calculation required {}s.".format(time.time()-start))

# Optimization of sigma-hyperparameter: 
sig_min=1.0e-2
sig_max=2.0e-1
N_sig = 20

start_t = time.time()
sig_vec, Delta_vec = [], []
for i in range(N_sig):
    Delta_f = 0
    for j in range(len(x)):
        Delta_f += (f_t(x[j])-f_ML(x[j],sig_min+i*(sig_max-sig_min)/(N_sig-1),Lambda))**2
    sig_vec.append(sig_min+i*(sig_max-sig_min)/(N_sig-1))
    Delta_vec.append(Delta_f)
    #print(i,sig_min+i*(sig_max-sig_min)/(N_sig-1),Delta_f)

### Check whether is it a local or global minima!

print("This calculation required {} min".format((time.time()-start_t)/60))

#plt.plot(sig_vec,np.log(Delta_vec))
#plt.plot()

sig_min = sig_vec[Delta_vec.index(min(Delta_vec))]
  
plt.plot(x,f_t(x),label='Original data')
plt.plot(x,f_ML(x,sig_min,Lambda),label='ML fit')
plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()


