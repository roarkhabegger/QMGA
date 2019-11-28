#QD Problem file for varational method genetic algorithm

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi as Ei

#Parameter infor for problem
NumPar = 1

#QD radius
R = 1.0

#search for values between 0 and SearchPar for lambda in wavefunction
SearchPar = 10

#Data processing after algorithm
def plotter(indLst):
    n = 2000
    plt.figure(1)
    rArr = np.linspace(R/(2*n),1.05*R,int(3*n/5))

    #plot all individuals and their minimized fitness
    for ind in indLst:
        par = ind[0]*SearchPar
        plt.plot(rArr,psi(rArr,par))
        plt.axhline(0)

        print(-1.0*Func(ind)[0]+10000)
    #parRange = np.arange(5.0, 10.0, 0.0001)
    f1Lst = np.zeros(len(parRange))
    plt.figure(2)
    for k in range(len(parRange)):
        f1Lst[k] = np.real(f1(parRange[k]))
    plt.plot(parRange,f1Lst)
    plt.show()


#Fitness evaluation function
def Func(ind):
    par = ind[0]*SearchPar
    val1 = 0.0
    #integrate the numerator and denominator
    val1 = f1(par)
    return (10000-val1,)

#Force parameter to be positive
def ParBounds(ind):
    if ind[0] > 0 :
        return True
    return False

#How invalid is the given individual
def DistOutOfBounds(ind):
    return 0.0-ind[0]

def psi(r,par):
    #parameters of model
    k = np.pi/R
    l = par
    #Normalization constant
    Norm = np.exp(l*R)*np.sqrt(l*(np.power(k,2)+np.power(l,2)))/np.sqrt(np.pi)
    Norm *= 1/np.sqrt((-1+np.exp(2*l*R))*np.power(k,2)-np.power(l,2)+
            np.power(l,2)*np.cos(2*k*R)-k*l*np.sin(2*k*R))

    #Value of wavefunction form at r
    val = np.sin(k*r)*np.exp(-1*l*r)*np.power(r,-1)

    #Return normalized phi
    return val * Norm



#Magnitude of approximation function (should be 1)
def f1(par1):
    a = -2.0j*np.pi-2.0*par1*R
    abar = 2.0j*np.pi-2.0*par1*R
    aRe = -2.0*par1*R
    val1 = (1.0-np.exp(-2.0*par1*R))/par1*(np.pi/R)**2
    val1 += 2.0*Ei(a)+2.0*Ei(abar)-4.0*Ei(aRe)
    val1 -= 2.0*np.log(1.0+(np.pi/(par1*R))**2)
    val2 = (1.0-np.exp(-2.0*par1*R))
    val2 *= (np.pi**2)/par1
    val2 *= ((np.pi**2)+(par1*R)**2)**(-1)

    return val1/val2
