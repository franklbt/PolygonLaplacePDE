# Solve Laplace's equation for u by iteration on an
# (N1+1)x(N2+1) grid in a polygon of radius 1.

import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

n = 6 #number of edges
N1 = 10
N2 = n*N1
EPS = 1.e-4


def main():

    dumax = 1. + EPS
    dR = 1./N1
    dTheta = 2* math.pi/N2

    # Initialize the grid.

    u = np.zeros((N1+1,N2+2))
    u[0:3*N1/4,0:N2+1] = 1

    # Apply boundary conditions.

    for i in range(n):
        for j in range((i)*N1,(i+1)*N1):
            if i%4 == 0 :
                u[N1,j] = math.sin(j*math.pi/(N2+1))
            elif i%4 == 2 : u[N1,j] = - math.sin(j*math.pi/(N2+1))
            else :u[N1,j] = 0

    # Show the initial conditions.

    def polygonPolarCoordinates(r,theta):
        #polar equation of a polygon with n edges
        return 2*r*np.sin(np.pi/n)/np.cos(theta - np.pi/n - (2*np.pi/n)*np.floor(theta*n/(2*np.pi)))

    R,T = np.meshgrid( [r/N1 for r in range(N1)],[theta*2* np.pi/N2 for theta in range(N2+1)] )

    X = polygonPolarCoordinates(R,T)*np.cos(T)
    Y = polygonPolarCoordinates(R,T)*np.sin(T)

    plt.title('Initial conditions')
    plt.contourf(X,Y, [[u[r][theta] for r in range(N1)] for theta in range(N2+1)])
    plt.colorbar()
    plt.xlabel('x index')
    plt.ylabel('y index')
    plt.pause(0.05)
    #--------------------------------------------------

    # Iterate to convergence.

    n_iter = 0
    while dumax > EPS:

        n_iter += 1
        uold = u.copy()

        # Numpy use of temporaries means that the following is
        # actually Jacobi, not Gauss-Seidel (slower convergence,
        # but *much* faster numpy computation).


        u[1:-1, 1:-1] = 0.25 * (u[0:-2, 1:-1] + u[2:, 1:-1]
                                 + u[1:-1,0:-2] + u[1:-1, 2:])

        u[ 1:-1, 0] =  0.25 * ( uold[0:-2, 0] + uold[2:, 0]
                                 + uold[1:-1,-1] + uold[1:-1, 1])
        u[ 1:-1, -1] =  0.25 * (uold[0:-2, -1] + uold[2:, -1]
                                 + uold[1:-1,-2] + uold[ 1:-1, 0])

        dumax = np.max(np.fabs(u-uold))

	#----------------------------------------------
        # Show the current iterate.
        
        plt.contourf(X,Y, [[u[r][theta] for r in range(N1)] for theta in range(N2+1)])
        plt.title('Jacobi convergence: iteration '+str(n_iter))
        plt.pause(0.05)
	#----------------------------------------------

    print( n_iter, 'iterations')
    plt.show()

if( __name__ == "__main__" ):
    main()
