import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def rhsfunc(x, phi, K, epsilon):   #define ODE
   f1 = phi[1]
   f2 = (K * (x**2) - epsilon) * phi[0]
   return np.array([f1, f2])

def shooting_method(L, K, xspan, y0, rhsfunc):  #returns the eigenvalues and the absolute value of the eigenfunctions 
   epsilon_start = 0   #initial epsilon value that we will change
   eigenvalues = np.zeros(5)    #initialize the arrays of eigenvalues and eigenfunctions
   eigenfunctions = np.zeros((5, 20 * L + 1))
   temp = y0

   for modes in range(5):   #finds the first five modes and their eigenvalues
      epsilon = epsilon_start   #set epsilon to the starting value
      depsilon = 1   #reset depsilon value
      y0 = temp

      for j in range(1000):    #performs the shooting method by changing epsilon and uses the bisection method to 
                               #find the eigenvalue and eigenfunction
         xevals = np.linspace(-L, L, 20 * L + 1)    #solves the ivp
         sol = scipy.integrate.solve_ivp(lambda x, phi: rhsfunc(x, phi, K, epsilon), xspan, y0, t_eval=xevals)
         y_sol = sol.y[0, :]
         
         if np.abs(np.sqrt(K * (L**2) - epsilon) * y_sol[-1] + sol.y[1, -1]) < 10**(-6):   #checks if the solution is within the desired tolerance of 10^(-6) and breaks the loop
            eigenvalues[modes] = epsilon
            break
         
         if (-1)**(modes)*(np.sqrt(K * (L**2) - epsilon) * y_sol[-1] + sol.y[1, -1]) > 0:   #implimentation of the bisection method
            epsilon = depsilon + epsilon
         else:
            epsilon = epsilon - (depsilon / 2)
            depsilon = depsilon / 2
         
         y0[1] = np.sqrt(K * (L**2) - epsilon)   #update initial condition
      
      epsilon_start = epsilon + 0.1   #increases epsilon to move on to finding the next mode

      norm = np.sqrt(np.trapz(np.multiply(y_sol, y_sol), sol.t))    #calculates the norm of phi
      func = y_sol / norm
      #plt.plot(sol.t, func, linewidth=3)   #plots the normalized eigenfunction
      
      eigenfunctions[modes] = np.abs(func)   #creates the array of the absolute value of the eigenfunctions

   eigenvalues = np.array([eigenvalues])   #creates the row vector of the eigenvalues
   
   return eigenvalues, eigenfunctions

L = 4   #define constants
K = 1
xspan = [-L, L]

y0 = np.array([1, np.sqrt(K * (L**2))])   #define initial condition

solshoot = shooting_method(L, K, xspan, y0, rhsfunc)   #call function
#plt.show()   #shows the plot of the eigefunctions

A1 = np.transpose(np.array([solshoot[1][0, :]]))   #formats the solutions and assigns them to the deliverable variables
A2 = np.transpose(np.array([solshoot[1][1, :]]))
A3 = np.transpose(np.array([solshoot[1][2, :]]))
A4 = np.transpose(np.array([solshoot[1][3, :]]))
A5 = np.transpose(np.array([solshoot[1][4, :]]))

A6 = solshoot[0]

# ------Presentation Problem------ #
from mpl_toolkits import mplot3d
from matplotlib import cm

x = np.linspace(-L, L, 20 * L + 1)
t = np.array([np.linspace(0, 5, 100)])

phi2 = np.array([solshoot[1][1]]).T
E2 = solshoot[0][0, 1]

psi2 = np.outer(phi2, np.cos((E2 / 2) * t))

fig1 = plt.figure()
ax1 = plt.axes(projection = '3d')
X, T = np.meshgrid(x, t)

ax1.plot_surface(X, T, psi2.T.real, cmap = cm.hsv)
ax1.set_xlabel('x axis')
ax1.set_ylabel('t axis')
ax1.set_zlabel('$\phi_2(x,t)$')
ax1.set_title('Time Evolution of the Second Mode')
plt.show()
# ------Presentation Problem------ #

def direct_method(L, K):   #returns the desired eigenvalues and eigenfunctions
   dx = (L * 2) / (20 * L)   #calculate dx
   center_diag = np.zeros(20 * L - 1)   #define arrays for the center diagonal and the offset diagonals
   offset_diag = np.ones(20 * L - 2) * (-1)    #inputs the correct values for the offset diagonals
   
   for i in range(20 * L - 1):   #inputs the correct values for x in the equation for the center diagonal
      center_diag[i] = 2 + K * ((-L + (i+1) * dx) ** 2) * (dx ** 2)
   
   A = np.diag(offset_diag, -1) + np.diag(offset_diag, 1) + np.diag(center_diag, 0)   #create the matrix A
   
   A[0, 0] = (2 / 3) + K * ((-L + dx) ** 2) * (dx ** 2)   #edit the starting and ending values of A to match the forward or backward difference schemes
   A[-1, -1] = (2 / 3) + K * ((L - dx) ** 2) * (dx ** 2)
   A[0, 1] = ((-2) / 3)   #edit the other values that were affected by using the forward and backward difference schemes
   A[-1, -2] = ((-2) / 3)
   
   tempeigenvals, tempeigenfuncs = np.linalg.eig(A)   #obtain the eigenvectors and eigenfuntions of A
   
   tempeigenvals = tempeigenvals * (1 / (dx ** 2))   #scale the eigenvalues by 1/dx**2
   
   indexlist = np.argsort(tempeigenvals)   #get the indices of the smallest eigenvalues
   
   eigenvals = np.zeros(5)   #initialize arrays for the eigenvalues and eigenvectors
   eigenfuncs = np.zeros((5, 79))
   
   for i in range(5):   #create the arrays for the five smallest eigenvalues and their corresponding eigenfunctions
      index = indexlist[i]
      eigenvals[i] = tempeigenvals[index]
      eigenfuncs[i] = tempeigenfuncs[:, index]
      
   eigenfuncs = np.transpose(eigenfuncs)   #transpose the eigenfunction array so that each column will be an eigenfunction
   
   phi0row = np.zeros(5)   #creates the row vectors that will hold all of the phi0 and phiN values
   phiNrow = np.zeros(5)
   
   for i in range(5):   #calculates phi0 and phiN values and creates vectors to hold them
      col = eigenfuncs[:, i]
      epsilon = eigenvals[i]
      
      phi0 = ((4 * col[0]) - col[1]) / ((2 * dx * np.sqrt(K * (L**2) - epsilon)) + 3)
      phiN = (((-4) * col[-1]) + col[-2]) / (((-2) * dx * np.sqrt(K * (L**2) - epsilon)) - 3)
      
      phi0row[i] = phi0
      phiNrow[i] = phiN
      
   solfuncs = np.zeros((20 * L + 1, 5))   #creates a new array that will hold the final eigenfunctions with the boundary values added
      
   for j in range(20 * L + 1):   #assembles the final eigenfunctions with their boundary values included
      if j == 0:
         solfuncs[0, :] = phi0row
      elif j == (20 * L):
         solfuncs[-1, :] = phiNrow
      else:
         solfuncs[j, :] = eigenfuncs[j-1, :]
      
   return eigenvals, solfuncs   #returns the eigenvalues of the matrix and the eigenfunctions with their boundary values added
   
eigenstuff = direct_method(L, K)  #call the function to get the eigenvalues and eigenfunctions

xlist = np.linspace(-4, 4, 20 * L + 1)
soldirect = np.zeros((5, 81))   #creates the array that will hold the normalized eigenfunctions
   
for i in range(5):   #normalize the eigenfunctions and adds them to the soldirect array. Also plots the first five eigenfunctions
   norm = np.sqrt(np.trapz(np.multiply(eigenstuff[1][:, i], eigenstuff[1][:, i]), xlist))
   soldirect[i] = eigenstuff[1][:, i] / norm
   #if i > 1:
      #plt.plot(xlist, (-1) * soldirect[i], linewidth=3)
   #else:
      #plt.plot(xlist, soldirect[i], linewidth=3)
#plt.show()

A7 = np.transpose(np.array([np.abs(soldirect[0])]))   #formats the solutions and assigns them to the deliverable variables
A8 = np.transpose(np.array([np.abs(soldirect[1])]))
A9 = np.transpose(np.array([np.abs(soldirect[2])]))
A10 = np.transpose(np.array([np.abs(soldirect[3])]))
A11 = np.transpose(np.array([np.abs(soldirect[4])]))

A12 = np.array([eigenstuff[0]])

def rhsfunc3(x, phi, K, epsilon, gamma):   #define ODE
   f1 = phi[1]
   f2 = ((gamma * (np.abs(phi[0]) ** 2)) + (K * (x ** 2)) - epsilon) * phi[0]
   return np.array([f1, f2])

def shooting_method3(L, K, xspan, y0, gamma, rhsfunc3):   #returns the eigenvalues and eigenfunctions for the ODE
   epsilon_start = 0   #initialize starting variables and arrays that will hold the eigenvalues and eigenfunctions
   eigenvalues = np.zeros(2)
   eigenfunctions = np.zeros((2, 20 * L + 1))
   xevals = np.linspace(-L, L, 20 * L + 1)
   A = 0.001   #define A value
   
   for modes in range(2):   #for loop over two modes
      epsilon = epsilon_start
      depsilon = 0.5
      
      for j in range(1000):   #for loop for shooting
         y0[0] = A   #update initial condition
         y0[1] = np.sqrt(K * (L**2) - epsilon) * A
         sol = scipy.integrate.solve_ivp(lambda x, phi: rhsfunc3(x, phi, K, epsilon, gamma), xspan, y0, t_eval=xevals)   #solves the ODE
         y_sol = sol.y[0, :]   #gets the y values for the solution
         
         norm = np.trapz(np.multiply(y_sol, y_sol), sol.t)   #compute norm
         BC = -np.sqrt(K * (L**2) - epsilon) * y_sol[-1]   #compute boundary condition
         
         if np.abs(np.trapz(np.multiply(y_sol, y_sol), sol.t) - 1) < 10 ** (-5) and np.abs(BC - sol.y[1, -1]) < 10 ** (-5):   #if norm and boundary condition met
            eigenvalues[modes] = epsilon   #adds epsilon to the array of eigenvalues
            break
         else:   #change A
            A = A / np.sqrt(norm)
         
         y0[0] = A   #update initial condition
         y0[1] = np.sqrt(K * (L**2) - epsilon) * A
         sol = scipy.integrate.solve_ivp(lambda x, phi: rhsfunc3(x, phi, K, epsilon, gamma), xspan, y0, t_eval=xevals)   #sovle ODE
         y_sol = sol.y[0, :]   #gets the y values for the solution
         
         norm = np.trapz(np.multiply(y_sol, y_sol), sol.t)   #compute norm
         BC = -np.sqrt(K * (L**2) - epsilon) * y_sol[-1]   #compute boundary condition
         
         if np.abs(np.trapz(np.multiply(y_sol, y_sol), sol.t) - 1) < 10 ** (-5) and np.abs(BC - sol.y[1, -1]) < 10 ** (-5):
            eigenvalues[modes] = epsilon   #adds epsilon to the array of eigenvalues
            break
         else:   #change epsilon according to the bisection method
            if (-1)**(modes)*(np.sqrt(K * (L**2) - epsilon) * y_sol[-1] + sol.y[1, -1]) > 0:
               epsilon = depsilon + epsilon
            else:
               epsilon = epsilon - (depsilon / 2)
               depsilon = depsilon / 2
      
      eigenfunctions[modes] = y_sol   #adds the normalized eigenfunction to the array of solutions
      #plt.plot(sol.t, y_sol, linewidth=3)   #plots the normalized eigenfunction
               
      epsilon_start = epsilon + 0.1   #incriments the starting epsilon value to move on to the next eigenvalue
      
   return eigenvalues, eigenfunctions   #returns the eigenvalues and normalized eigenfunctions

L = 3   #define constants
K = 1
gamma = 0.05
xspan = [-L, L]

y0 = np.array([0.001, np.sqrt(K * (L**2)) * 0.001])   #define initial condition

sol1shoot3 = shooting_method3(L, K, xspan, y0, gamma, rhsfunc3)   #call function
#plt.show()

A13 = np.transpose(np.array([np.abs(sol1shoot3[1][0])]))   #formats the solutions and assigns them to the deliverable variables
A14 = np.transpose(np.array([np.abs(sol1shoot3[1][1])]))

A15 = np.array([sol1shoot3[0]])

gamma = -0.05   #changes gamma to -0.05
sol2shoot3 = shooting_method3(L, K, xspan, y0, gamma, rhsfunc3)   #call function
plt.show()

A16 = np.transpose(np.array([np.abs(sol2shoot3[1][0])]))   #formats the solutions and assigns them to the deliverable variables
A17 = np.transpose(np.array([np.abs(sol2shoot3[1][1])]))

A18 = np.array([sol2shoot3[0]])