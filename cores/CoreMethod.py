import math
import numpy as np


# --------------------------------------------------------------------------------- #


def InitializeVariables(nMax):
    # primitive variables
    X = np.zeros(nMax)
    A = np.zeros(nMax)
    p = np.zeros(nMax)
    rho = np.zeros(nMax)
    V = np.zeros(nMax)
    T = np.zeros(nMax)
    mDot = np.zeros(nMax)
    M = np.zeros(nMax)

    # compact variables
    U = np.zeros((nMax, 3))
    Up = np.zeros((nMax, 3))
    F = np.zeros((nMax, 3))
    J = np.zeros((nMax-1, 3))
    S = np.zeros((nMax-1, 3))
    dUp = np.zeros((nMax-1, 3))
    dUc = np.zeros((nMax-1, 3))
    dUav = np.ones((nMax-1, 3))    

    return (X, A, p, rho, V, T, mDot, M, U, Up, F, J, S, dUp, dUc, dUav)

def GenerateInitialCondition(X, A, rho, T, p, V, M, mDot, U,
                            nMax, L, rhoi, Ti, pi, pe, gamma):
    
    # Initialize: Geometry value
    #   Dicretization
    dx = L/(nMax-1)
    #   X and Area
    for i in range(nMax):
        X[i] = i*dx
        A[i] = 1 + 2.2*((X[i] - 1.5)**2)

    # Initialize: Computational variables
    #   Primitive variables: rho, T, p, V, M, mDot
    #   Compact variable: U
    for i in range(nMax):
        rho[i] = rhoi
        T[i] = Ti

        if (pe==1.0):
            # if (pe = 1), it means there is no flow.
            # end the calculation after initialize initial condition
            p[i] = 1.0
            V[i] = 0.0
            M[i] = 0.0
            mDot[i] = 0.0
        else:
            # it means, there is flow throughout nozzle
            p[i] = rho[i]*T[i]
            mDot[i] = 0.579
            V[i] = mDot[i]/(rho[i]*A[i])
            M[i] = V[i]/(T[i]**0.5)

            U[i,0] = rho[i]*A[i]
            U[i,1] = rho[i]*A[i]*V[i]
            U[i,2] = rho[i]*((T[i]/(gamma-1)) + (0.5*gamma*((V[i])**2)))*A[i]

    return (dx, X, A, rho, T, p, V, M, mDot, U)                           

def GenerateTimeStep(C, dx, T, V, nMax):
    dt = 1.0

    # Calculate time step
    for i in range(nMax-1):
        tTemp = C*dx/(np.sqrt(T[i]) + V[i])
        if (tTemp < dt):
            dt = tTemp

    return (dt)

def CalculateCVF(F, U, gamma, nMax):
    for i in range(nMax):
        F[i,0] = U[i,1]
        F[i,1] = ((U[i,1]**2)/U[i,0]) + (((gamma-1.0)/gamma)*(U[i,2] - 0.5*gamma*((U[i,1]**2)/U[i,0])))
        F[i,2] = (gamma*U[i,1]*U[i,2]/U[i,0]) - (0.5*gamma*(gamma-1.0)*((U[i,1]**3)/(U[i,0]**2)))

    return (F)

def PredictorLoop(U, Up, dUp, F, J, S, A, rho, T, p, V, gamma, dt, dx, C_x, pe, nMax):
    F = CalculateCVF(F, U, gamma, nMax)    # Compact Variables: F
    
    for i in range(1,nMax-1):
        Q = C_x*np.abs(p[i+1] - 2*p[i] + p[i-1])/(p[i+1] + 2*p[i] + p[i-1])     # Factor for Aritificial Viscosity

        # Compact Variables: J
        J[i,0] = 0.0
        J[i,1] = ((gamma-1.0)/gamma)*(U[i,2] - (0.5*gamma*((U[i,1]**2)/U[i,0])))*(np.log(A[i+1])-np.log(A[i]))/dx
        J[i,2] = 0.0

        for j in range(3):
            S[i,j] = Q*(U[i+1,j] - 2*U[i,j] + U[i-1,j])     # Artificial Dissipation
            dUp[i,j] = -(F[i+1,j] - F[i,j])/dx + J[i,j]     # Gradient of intermediate solution
            Up[i,j] = U[i,j] + dUp[i,j]*dt + S[i,j]         # Intermediate solution: U

    for i in range(nMax):
        # B.C. correction
        # at inlet
        if (i==0):
            for j in range(3):
                Up[i,j] = U[i,j]
        # at outlet
        elif (i==nMax-1):
            for j in range(3):
                Up[i,j] = U[nMax-1,j]

            # if shockwave appears (isentropic flows occurs if pe = 0.0)
            if (pe > 0.01):
                Up[nMax-1,2] = (pe*A[nMax-1]/(gamma-1)) + 0.5*gamma*Up[nMax-1,1]*V[nMax-1]

        # Recalculate primitive variables        
        rho[i] = Up[i,0]/A[i]
        T[i] = (gamma-1)*((Up[i,2]/Up[i,0]) - (0.5*gamma*(V[i]**2)))
        p[i] = rho[i]*T[i]
        V[i] = Up[i,1]/Up[i,0]
        
    # Intermediate solution: F
    F = CalculateCVF(F, Up, gamma, nMax)

    return (U, Up, dUp, F, J, S, A, rho, T, p, V)

def CorrectorLoop(U, Up, dUp, dUc, dUav, F, J, S, A, rho, T, p, V, gamma, dt, dx, C_x, pe, nMax):
    for i in range(1,nMax-1):
        Q = C_x*np.abs(p[i+1] - 2*p[i] + p[i-1])/(p[i+1] + 2*p[i] + p[i-1])     # Factor for Aritificial Viscosity

        # Compact Variables: J
        J[i,0] = 0.0
        J[i,1] = ((gamma-1.0)/gamma)*(Up[i,2] - (0.5*gamma*((Up[i,1]**2)/Up[i,0])))*(np.log(A[i])-np.log(A[i-1]))/dx
        J[i,2] = 0.0

        for j in range(3):
            S[i,j] = Q*(Up[i+1,j] - 2*Up[i,j] + Up[i-1,j])      # Artificial Dissipation
            dUc[i,j] = -(F[i,j] - F[i-1,j])/dx + J[i,j]         # Gradient of corrector solution
            dUav[i,j] = 0.5*(dUp[i,j] + dUc[i,j])               # Gradient of average solution
            U[i,j] = U[i,j] + dUav[i,j]*dt + S[i,j]             # Final solution: U

    # B.C. correction
    U[0,0] = rho[0]*A[0]  
    U[0,1] = 2*U[1,1] - U[2,1]
    U[0,2] = U[0,0]*((T[0]/(gamma-1)) + (0.5*gamma*(V[0]**2)))

    for j in range(3):
        U[nMax-1,j] = 2*U[nMax-2,j] - U[nMax-3,j]

    # if shockwave appears (isentropic flows occurs if pe = 0.0)  
    if (pe>0.01):
        U[nMax-1, 2] = (pe*A[nMax-1]/(gamma-1)) + 0.5*gamma*Up[nMax-1,1]*V[nMax-1]

    # Recalculate primitive variables  
    for i in range(nMax):        
        rho[i] = Up[i,0]/A[i]
        T[i] = (gamma-1)*((Up[i,2]/Up[i,0]) - (0.5*gamma*(V[i]**2)))
        p[i] = rho[i]*T[i]
        V[i] = Up[i,1]/Up[i,0]

    return (U, dUav, rho, T, p, V)

def WriteResults(rho, T, V, p, mDot, M, X, A, nMax):
    title = "numeric-results.csv"
    fname = "results/" + title
    
    f = open(fname, "w+")
    f.write("X, A, rho, T, V, p, mDot, M")
    f.write("\n")
    for i in range(nMax):
        f.write(f"{np.round(X[i],3)}, {np.round(A[i],3)}, {np.round(rho[i],3)},")
        f.write(f"{np.round(T[i],3)}, {np.round(V[i],3)}, {np.round(p[i],3)},")
        f.write(f"{np.round(mDot[i],3)}, {np.round(M[i],3)}")
        if (i!=(nMax-1)):
            f.write("\n")
    f.close()

    title = "coordinates.csv"
    fname = "results/" + title

    r = np.zeros(nMax)
    for i in range(nMax):
        r[i] = np.sqrt(A[i]/math.pi)

    f = open(fname, "w+")
    for i in range(nMax):
        f.write(f"{1} {i+1} {np.round(X[i],3)} {np.round(r[i],3)} {0}")
        if (i!=(nMax-1)):
            f.write("\n")
    f.close()

# --------------------------------------------------------------------------------- #