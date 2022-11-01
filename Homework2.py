import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

def rhsfunc(x, phi, K, epsilon):   #define ODE
   f1 = phi[1]
   f2 = (K * (x**2) - epsilon) * phi[0]
   return np.array([f1, f2])

L = 4   #define constants
K = 1
xspan = [-L, L]

#define initial condition
y0 = np.array([1, 1])

def shooting_method(L, K, xspan, y0, rhsfunc):  #returns the eigenvalues and the absolute value of the eigenfunctions 
   epsilon_start = 0   #initial epsilon value that we will change
   eigenvalues = np.zeros(5)    #initialize the array of eigenvalues and eigenfunctions
   eigenfunctions = np.zeros((5, 20 * L + 1))

   for modes in range(5):   #finds the first five modes and their eigenvalues
      epsilon = epsilon_start   #reset epsilon to the starting value
      depsilon = 1   #reset depsilon value

      for j in range(1000):    #performs the shooting method by changing epsilon and uses the bisection method to 
                               #find the eigenvalue and eigenfunction
         xevals = np.linspace(-L, L, 20 * L + 1)    #solves the ivp
         sol = scipy.integrate.solve_ivp(lambda x, phi: rhsfunc(x, phi, K, epsilon), xspan, y0, t_eval=xevals)
         y_sol = sol.y[0, :]
         
         if np.abs(y_sol[-1]) < 10**(-6):   #checks if the solution is within the desired tolerance of 10^(-6) and breaks the loop
            eigenvalues[modes] = epsilon
            break
         
         if (-1)**(modes)*y_sol[-1] > 0:   #implimentation of the bisection method
            epsilon = depsilon + epsilon
         else:
            epsilon = epsilon - (depsilon / 2)
            depsilon = depsilon / 2
      
      epsilon_start = epsilon + 0.1   #increases epsilon to move on to finding the next mode

      norm = np.sqrt(np.trapz(np.multiply(y_sol, y_sol), sol.t))    #calculates the norm of phi
      #plt.plot(sol.t, sol.y[0, :] / norm, linewidth=3)   #plots the normalized eigenfunction
      
      eigenfunctions[modes] = np.abs(y_sol)   #creates the array of the absolute value of the eigenfunctions

   eigenvalues = np.array([eigenvalues])   #creates the row vector of the eigenvalues
   
   return eigenvalues, eigenfunctions

#the second for loop is not breaking and printing the eigenvalues when L = 6
#this means that the y_sols are not within the required error for the problem

solshoot = shooting_method(L, K, xspan, y0, rhsfunc)

A1 = np.transpose(np.array([solshoot[1][0, :]]))   #formats each eigenfunction into a column vector
A2 = np.transpose(np.array([solshoot[1][1, :]]))
A3 = np.transpose(np.array([solshoot[1][2, :]]))
A4 = np.transpose(np.array([solshoot[1][3, :]]))
A5 = np.transpose(np.array([solshoot[1][4, :]]))

A6 = solshoot[0]

#plt.show()   #shows the plot of the eigefunctions

def direct_method(L, K):   #returns the desired eigenvalues and eigenfunctions
   dx = (L * 2) / (20 * L + 1)   #calculate dx
   center_diag = np.zeros(20 * L - 1)   #define arrays for the center diagonal and the offset diagonals
   offset_diag = np.ones(20 * L - 2) * (1 / (dx ** 2))   #inputs the correct values for the offset diagonals
   
   for i in range(20 * L - 1):   #inputs the correct values for the center diagonal
      center_diag[i] = ((-2) / (dx ** 2)) - K * ((-L + ((i+1) * dx)) ** 2)
   
   A = np.diag(offset_diag, -1) + np.diag(offset_diag, 1) + np.diag(center_diag, 0)   #create the matrix A
   
   A[0, 0] = ((-2) / (3 * (dx ** 2))) - K * ((-L + dx) ** 2)   #edit the starting and ending values of A to match the forward or backward difference schemes
   A[-1, -1] = ((-2) / (3 * (dx ** 2))) - K * ((L - dx) ** 2)
   A[0, 1] = 2 / (3 * (dx ** 2))   #edit the other values that were affected by using the forward and backward difference schemes
   A[-1, -2] = 2 / (3 * (dx ** 2))
   
   return np.linalg.eig(A)   #return the eigenvectors and eigenfuntions of A
   
eigenstuff = direct_method(L, K)  #call the function to get the eigenvalues and eigenfunctions

xlist = np.linspace(-4, 4, 20 * L - 1)
   
for i in range(5):   #plot the first five eigenfunctions
   norm = np.sqrt(np.trapz(np.multiply(eigenstuff[1][:, i], eigenstuff[1][:, i]), xlist))
   plt.plot(xlist, eigenstuff[1][:, i] / norm)
plt.show()

def rhsfunc3(x, phi, K, epsilon, gamma):   #define ODE
   f1 = phi[1]
   f2 = ((gamma * (np.abs(phi[0]) ** 2)) + (K * (x ** 2)) - epsilon) * phi[0]
   
L = 3   #define constants
K = 1
gamma = 0.05
xspan = np.linspace(-L, L, 20 * L + 1)

y0 = np.array([1, 1])   #define initial condition

def shooting_method3(L, K, xspan, y0, gamma, rhsfunc3):
   