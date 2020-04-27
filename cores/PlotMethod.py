import math
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------- #


def PlotArea(X, A, nMax, L, title="Chart Area"):
    # generate plot
    r = np.zeros(nMax)
    for i in range(nMax):
        r[i] = np.sqrt(A[i]/math.pi)

    fig = plt.figure(figsize=(13.5,4.5))
    plt.plot(X, r, 'r', label=r'$A(x) = 1 + 2.2(x-1.5)^{2}$')
    plt.plot(X, -r, 'r')
    plt.xlabel(r'$X^{*}$')
    plt.ylabel(r"$r(x)$")
    plt.legend(prop={"size":10}, loc='upper center')
    plt.xlim(0, L)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')

    plt.show()      # show plot

    # save plot
    fName = "results/" + title + ".png"
    fig.savefig(fName, dpi = 150)

def PlotResidue(numiter, resRho, resT, resV, resU, maxIter, title="Residual"):
    fig = plt.figure(figsize=(9, 6))
    plt.subplot(211)
    plt.plot(numiter, resRho, color='r', linestyle='-', linewidth='1', label=r'$|d\rho/dt|_{av}$')
    plt.plot(numiter, resT, color='g', linestyle='--', linewidth='1', label=r'$|dT/dt|_{av}$')
    plt.plot(numiter, resV, color='b', linestyle='-.', linewidth='1', label=r'$|dV/dt|_{av}$')
    plt.legend(prop={"size":10}, loc='upper right')
    plt.xlim(1, maxIter)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.ylabel("Residuals")
    #plt.yscale("log")

    plt.subplot(212)
    plt.plot(numiter, resU[:,0], color='r', linestyle='-', linewidth='1', label=r'$|dU_{1}/dt|_{av}$')
    plt.plot(numiter, resU[:,1], color='g', linestyle='--', linewidth='1', label=r'$|dU_{2}/dt|_{av}$')
    plt.plot(numiter, resU[:,2], color='b', linestyle='-.', linewidth='1', label=r'$|dU_{3}/dt|_{av}$')
    plt.legend(prop={"size":10}, loc='upper right')
    plt.xlim(1, maxIter)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.ylabel("Residuals")
    plt.xlabel("Timesteps")
    #plt.yscale("log")

    plt.show()   

    # save plot
    fName = "results/" + title + ".png"
    fig.savefig(fName, dpi = 150)

def PlotPrimitiveVariables(X, A, rho, T, V, L, nMax, title="Primitive Variables"):
    r = np.zeros(nMax)
    for i in range(nMax):
        r[i] = np.sqrt(A[i]/math.pi)

    fig = plt.figure(figsize=(9,8))

    plt.subplot(411)
    plt.plot(X, r, color='b', linewidth='2', label=r'$A(x) = 1 + 2.2(x-1.5)^{2}$')
    plt.plot(X, -r, color='b', linewidth='2')
    plt.legend(prop={"size":10}, loc='center right')
    plt.xlim(0, L)
    plt.ylabel(r"$r(x)$")
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')

    plt.subplot(412)
    plt.plot(X, rho, color='r', linestyle='-', linewidth='1', label="Numerical Solution")
    plt.ylabel(r'$\rho*$')   
    plt.legend(prop={"size":10}, loc='upper right')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.xlim(0.0, L)

    plt.subplot(413)
    plt.plot(X, T, color='r', linestyle='-', linewidth='1')
    plt.ylabel(r'$T*$')   
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.xlim(0.0, L)

    plt.subplot(414)
    plt.plot(X, V, color='r', linestyle='-', linewidth='1')
    plt.ylabel(r'$V*$')   
    plt.xlabel(r'$X*$')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.xlim(0.0, L)

    plt.show()    

    # save plot
    fName = "results/" + title + ".png"
    fig.savefig(fName, dpi = 150)

def PlotMDot(X, mDot, L, nMax, title="Mass Flow Comparison"):
    fig = plt.figure(figsize=(13.5,4.5))

    plt.plot(X, mDot, color='r', linestyle='-', linewidth='1', label="Numerical solution")
    plt.plot(X, 0.579*np.ones(nMax), color='k', linestyle='--', linewidth='2', label='exact solution')
    plt.ylabel(r'$(\rho A V)^{*}$')
    plt.xlabel(r'$X^{*}$')   
    plt.legend(prop={"size":10}, loc='upper right')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.xlim(0.0, L)

    plt.show()    

    # save plot
    fName = "results/" + title + ".png"
    fig.savefig(fName, dpi = 150)   


# ---------------------------------------------------------------------------- #