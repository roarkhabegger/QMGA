#ProbFile for Infinite Well
import numpy as np
import matplotlib.pyplot as plt

NumPar = 1

hbarc = 0.19732698*10**(-6) # eV m
mc2 = 0.51099895*10**(6) # eV
l = 0.21 #m
wDivc = 2.0*np.pi/l
a = 1.0
SearchPar = 4.0* mc2*wDivc/hbarc

def plotter(indLst):
    n = 3000
    plt.figure(1,figsize=(10,6))
    rArr = np.linspace(-10.0/SearchPar,10/SearchPar,n)
    for ind in indLst:
        par = ind[0]*SearchPar
        plt.plot(rArr*0.25*SearchPar,psi(rArr,par)/np.sqrt(0.25*SearchPar))
        ylabStr = "$\psi(x)$ \n $[\hbar m^{-1} \omega^{-1}]^{0.5}$"
        plt.ylabel(ylabStr,rotation = 0,labelpad = 25, position = (0,0.45))
        plt.xlabel("$x$ $[m \omega \hbar^{-1}]$")
        plt.xticks(np.arange(-2,2.1,1))
        plt.title("Harmonic Oscillator Minimum Energy $\psi$")

        #plt.axhline(0)
        print(par)
        print(mc2*wDivc/hbarc)
        print(hbarc*wDivc/2)
        print(100*hbarc*wDivc-Func(ind)[0])

    #plt.show()

def Func(ind):
    #par = [0.0,0.0]
    #par[0] = ind[0]*SearchPar[0]
    par = ind[0]*SearchPar
    val = (hbarc**2)/(4*mc2)*par + mc2*(wDivc**2)/(4.0*par)

    return (100*hbarc*wDivc-val,)

def ParBounds(ind):
    #print(ind)
    if ind[0] > 0:
        return True

    return False

def DistOutOfBounds(ind):
    return 0.0-ind[0]

def psi(r,params):
    par = params*SearchPar
    Norm = np.sqrt(np.sqrt(par/np.pi))
    return Norm*np.exp(-1.0*par/2*np.power(r,2))
