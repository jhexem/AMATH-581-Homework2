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
      func = y_sol / norm
      #plt.plot(sol.t, y_sol / norm, linewidth=3)   #plots the normalized eigenfunction
      
      eigenfunctions[modes] = np.abs(func)   #creates the array of the absolute value of the eigenfunctions

   eigenvalues = np.array([eigenvalues])   #creates the row vector of the eigenvalues
   
   return eigenvalues, eigenfunctions

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
   offset_diag = np.ones(20 * L - 2) * (-1)    #inputs the correct values for the offset diagonals
   
   for i in range(20 * L - 1):   #inputs the correct values for the center diagonal
      center_diag[i] = 2 + K * ((-L + (i+1) * dx) ** 2) * (dx ** 2)
   
   A = np.diag(offset_diag, -1) + np.diag(offset_diag, 1) + np.diag(center_diag, 0)   #create the matrix A
   
   A[0, 0] = (2 / 3) + K * ((-L + dx) ** 2) * (dx ** 2)   #edit the starting and ending values of A to match the forward or backward difference schemes
   A[-1, -1] = (2 / 3) + K * ((L - dx) ** 2) * (dx ** 2)
   A[0, 1] = ((-2) / 3)   #edit the other values that were affected by using the forward and backward difference schemes
   A[-1, -2] = ((-2) / 3)
   
   eigenvals, eigenfuncs = np.linalg.eig(A)   #obtain the eigenvectors and eigenfuntions of A
   
   epsilonfirst = eigenvals[0]   #gets the first and last eigenvalues to calculate phi0 and phiN
   epsilonlast = eigenvals[-1]
   
   phi0row = np.zeros(20 * L -1)   #creates the row vectors that will hold all of the phi0 and phiN values
   phiNrow = np.zeros(20 * L -1)
   
   for i in range(20 * L -1):   #calculates phi0 and phiN values and adds them to their row vectors
      col = eigenfuncs[:, i]
      
      phi0 = ((4 * col[0]) - col[1]) / ((2 * (dx ** 3) * np.sqrt(K * (L**2) - epsilonfirst)) + 3)
      phiN = (((-4) * col[-1]) + col[-2]) / (((-2) * (dx ** 3) * np.sqrt(K * (L**2) - epsilonlast)) - 3)
      
      phi0row[i] = phi0
      phiNrow[i] = phiN
      
   solfuncs = np.zeros((20 * L + 1, 20 * L - 1))   #creates a new array that will hold the final eigenfunctions with the boundary values
      
   for j in range(20 * L + 1):   #assembles the final eigenfunctions with their boundary values included
      if j == 0:
         solfuncs[0, :] = phi0row
      elif j == (20 * L):
         solfuncs[-1, :] = phiNrow
      else:
         solfuncs[j, :] = eigenfuncs[j-1, :]
         
   eigenvals = eigenvals * (1 / ((L * 2) / (20 * L + 1)) ** 2)
      
   return eigenvals, solfuncs   #returns the eigenvalues of the matrix and the eigenfunctions with their boundary values added
   
eigenstuff = direct_method(L, K)  #call the function to get the eigenvalues and eigenfunctions

xlist = np.linspace(-4, 4, 20 * L + 1)
soldirect = np.zeros((5, 81))
   
for i in range(5):   #plot the first five eigenfunctions
   norm = np.sqrt(np.trapz(np.multiply(eigenstuff[1][:, i+61], eigenstuff[1][:, i+61]), xlist))
   soldirect[i] = eigenstuff[1][:, i+61] / norm
   '''if i % 2 == 0:
      plt.plot(xlist, soldirect[i], linewidth=3)
   else:
      plt.plot(xlist, (-1) * soldirect[i], linewidth=3)
plt.show()'''

A7 = np.transpose(np.abs(soldirect[0]))
A8 = np.transpose(np.abs(soldirect[1]))
A9 = np.transpose(np.abs(soldirect[2]))
A10 = np.transpose(np.abs(soldirect[3]))
A11 = np.transpose(np.abs(soldirect[4]))

A12 = eigenstuff[0][61:66]

def rhsfunc3(x, phi, K, epsilon, gamma):   #define ODE
   f1 = phi[1]
   f2 = ((gamma * (np.abs(phi[0]) ** 2)) + (K * (x ** 2)) - epsilon) * phi[0]
   return np.array([f1, f2])
   
L = 3   #define constants
K = 1
gamma = 0.05
xspan = [-L, L]
A = 0.01

y0 = np.array([A, 1])   #define initial condition

def shooting_method3(L, K, xspan, y0, gamma, rhsfunc3):
   epsilon_start = 0
   eigenvalues = np.zeros(2)
   eigenfunctions = np.zeros((2, 20 * L + 1))
   A = y0[0]
   xevals = np.linspace(-L, L, 20 * L + 1)
   
   for modes in range(2):   #for loop over two modes
      epsilon = epsilon_start
      depsilon = 0.5
      
      for j in range(1000):   #for loop for shooting
         sol = scipy.integrate.solve_ivp(lambda x, phi: rhsfunc3(x, phi, K, epsilon, gamma), xspan, y0, t_eval=xevals)   #solves the ODE
         y_sol = sol.y[0, :]
         
         norm = np.sqrt(np.trapz(np.multiply(y_sol, y_sol), sol.t))   #compute norm
         BC = y_sol[-1]   #compute boundary condition
         
         if np.abs(norm - 1) < 10 ** (-5) and np.abs(BC - A) < 10 ** (-5):   #if norm and boundary condition met
            eigenvalues[modes] = epsilon
            break
         else:   #change A
            A = A / np.sqrt(norm)
         
         y0 = np.array([A, 1])   #update initial condition
         sol = scipy.integrate.solve_ivp(lambda x, phi: rhsfunc3(x, phi, K, epsilon, gamma), xspan, y0, t_eval=xevals)   #sovle ODE
         y_sol = sol.y[0, :]
         
         norm = np.sqrt(np.trapz(np.multiply(y_sol, y_sol), sol.t))   #compute norm
         BC = y_sol[-1]
         
         if np.abs(norm - 1) < 10 ** (-5) and np.abs(BC - A) < 10 ** (-5):
            eigenvalues[modes] = epsilon
            break
         else:   #change epsilon
            if (-1)**(modes)*y_sol[-1] > 0:   #bisection method
               epsilon = depsilon + epsilon
            else:
               epsilon = epsilon - (depsilon / 2)
               depsilon = depsilon / 2
      
      norm = np.sqrt(np.trapz(np.multiply(y_sol, y_sol), sol.t))
      func = y_sol / norm
      eigenfunctions[modes] = func
      plt.plot(sol.t, func)
               
      epsilon_start = epsilon + 0.1
      
   return eigenvalues, eigenfunctions

solshoot3 = shooting_method3(L, K, xspan, y0, gamma, rhsfunc3)
print(solshoot3[0])
plt.show()