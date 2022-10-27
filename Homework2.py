import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

#define ODE
def rhsfunc(x, phi, K, epsilon):
   f1 = phi[1]
   f2 = (K * (x**2) - epsilon) * phi[0]
   return np.array([f1, f2])

#define constants
L = 4
x = L
K = 1
xspan = [-L, L]

#define initial conditions
A = 1   #shooting method parameter that we will change

tol = 10**(-6)

y0 = np.array([1, A])
epsilon_start = 0   #initial beta value that we will change

for modes in range(5):

   epsilon = epsilon_start
   depsilon = 2

   for j in range(1000):
      #solve the ivp
      xevals = np.linspace(-L, L, 20 * L + 1)
      sol = scipy.integrate.solve_ivp(lambda x, phi: rhsfunc(x, phi, K, epsilon), xspan, y0, t_eval=xevals)
      y_sol = sol.y[0, :]
      
      if np.abs(y_sol[-1]) < tol:
         print(epsilon)
         break
      
      if (-1)**(modes)*y_sol[-1] > 0:
         epsilon = depsilon + epsilon
      else:
         epsilon = epsilon - (depsilon / 2)
         depsilon = depsilon / 2
         
   epsilon_start = epsilon + 0.1

   norm = np.sqrt(np.trapz(np.multiply(y_sol, y_sol), sol.t))
   plt.plot(sol.t, sol.y[0, :] / norm, linewidth=3)

plt.show()