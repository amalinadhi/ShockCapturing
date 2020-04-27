import copy
import numpy as np
import os

import cores.CoreMethod as CoreMethod
import cores.PlotMethod as PlotMethod


# --------------------------------------------------------------------------------- #

# Directory
# Create directory
dirName = "results"

if not os.path.exists(dirName):
    os.mkdir(dirName)
else:
    pass

# --------------------------------------------------------------------------------- #

# --- INPUT: Discretization --- #
nMax = 121

# --- INITIALIZE VARIABLES --- #
X, A, p, rho, V, T, mDot, M, U, Up, F, J, S, dUp, dUc, dUav = CoreMethod.InitializeVariables(nMax)

# --- INPUT: Physical Condition --- #
# Geometry
L = 3.0         # Length of Nozzles

# Define boundary condition
# All the total properties is calculated in inlet condition
rhoi = 1.0      # non-dimensional inlet static density (rho/rho_t)
Ti = 1.0        # non-dimensional inlet static temperature (T/T_t)
pi = rhoi*Ti    # non-dimensional inlet static pressure (p inlet/p_t)
pe = 0.6784     # non-dimensional exit static pressure (p exit/p_t)

# Constant
gamma = 1.4     # ratio of specific heat capacity

# --- INITIALIZATION --- #
dx, X, A, rho, T, p, V, M, mDot, U = CoreMethod.GenerateInitialCondition(X, A, rho, T, p, V, M, mDot, U,
                                                                        nMax, L, rhoi, Ti, pi, pe, gamma)

# if there is flow (pe != 1), start calculating
if (pe != 1):
    # --- INPUT: MacCormack Iteration & Artificial Viscosity --- #
    # Artifical viscosity input
    C_x = 0.2               # Artificial viscosity constant

    # Time step input
    C = 0.5                 # Courant number

    # Iteration input
    maxIter = 3400          # number of iteration

    # residual variables
    numiter = np.zeros(maxIter)
    resRho = np.ones(maxIter)
    resT = np.ones(maxIter)
    resV = np.ones(maxIter)
    resU = np.ones((maxIter, 3))

    # Start iteration
    for iters in range(maxIter):
        # resiudal temporary storage
        rho_old = copy.deepcopy(rho)
        V_old = copy.deepcopy(V)
        T_old = copy.deepcopy(T)
        U_old = copy.deepcopy(U)

        # Generate timestep
        dt = CoreMethod.GenerateTimeStep(C, dx, T, V, nMax)
        
        # Predictor loop
        U, Up, dUp, F, J, S, A, rho, T, p, V = CoreMethod.PredictorLoop(U, Up, dUp, F, J, S, A, rho, 
                                                                        T, p, V, gamma, dt, dx, C_x, 
                                                                        pe, nMax)
        
        # Corrector loop
        U, dUav, rho, T, p, V = CoreMethod.CorrectorLoop(U, Up, dUp, dUc, dUav, F, J, S, A, rho, T, 
                                                            p, V, gamma, dt, dx, C_x, pe, nMax)

        # Generate Residual
        numiter[iters] = iters+1
        resRho[iters] = abs(np.sum(rho) - np.sum(rho_old))/np.sum(rho_old)
        resT[iters] = abs(np.sum(T) - np.sum(T_old))/np.sum(T_old)
        resV[iters] = abs(np.sum(V) - np.sum(V_old))/np.sum(V_old)
        resU[iters,0] = abs(np.sum(U[:,0]) - np.sum(U_old[:,0]))/np.sum(U_old[:,0])
        resU[iters,1] = abs(np.sum(U[:,1]) - np.sum(U_old[:,1]))/np.sum(U_old[:,1])
        resU[iters,2] = abs(np.sum(U[:,2]) - np.sum(U_old[:,2]))/np.sum(U_old[:,2])


    for i in range(nMax):
        mDot[i] = rho[i]*A[i]*V[i]
        M[i] = V[i]/(np.sqrt(T[i]))



# Post processing
# Plotting
PlotMethod.PlotArea(X, A, nMax, L, title="Chart Area")
PlotMethod.PlotResidue(numiter, resRho, resT, resV, resU, maxIter, title="Residual")
PlotMethod.PlotPrimitiveVariables(X, A, rho, T, V, L, nMax, title="Primitive Variables")
PlotMethod.PlotMDot(X, mDot, L, nMax, title="Mass Flow Comparison")

# Write results
CoreMethod.WriteResults(rho, T, V, p, mDot, M, X, A, nMax)
