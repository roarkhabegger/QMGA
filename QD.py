#QD Problem file for varational method genetic algorithm

import numpy as np
import matplotlib.pyplot as plt

#Parameter infor for problem
NumPar = 1

#QD radius
R = 2.0

#search for values between 0 and SearchPar for lambda in wavefunction
SearchPar = 20

#Runga-Kutta 4th order time integrator
def rk4(x0,y0,df,h,params):
    k1 = df(x0, params)*h
    k2 = df(x0+h/2.0, params)*h
    k3 = df(x0+h/2.0, params)*h
    k4 = df(x0+h, params)*h
    return (k1 + 2.*k2 + 2.*k3 + k4)/6.0+y0

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
        print(par)
        print(10000-FuncPrecise(ind)[0])
    plt.show()

#Fitness evaluation function
def Func(ind):
    par = ind[0]*SearchPar
    n = 2000
    x0 = R/n
    #Resolve near 0 more than at R
    xArr = np.linspace(x0,R*1.01,n)
    val1 = 0.0
    val2 = 0.0
    #integrate the numerator and denominator
    for i in range(n-1):
        val1 = rk4(xArr[i],val1,f1,xArr[i+1]-xArr[i],par)
        val2 = rk4(xArr[i],val2,f2,xArr[i+1]-xArr[i],par)

    return (10000-val2/val1,)

#Force parameter to be positive
def ParBounds(ind):
    if ind[0] > 0 :
        return True
    return False

#How invalid is the given individual
def DistOutOfBounds(ind):
    return 0.0-ind[0]

#approximation Wavefuntion (the test function)
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

#derivative of wavefunction
def dpsi(r,par):
    #parameters of model
    k = np.pi/R
    l = par
    #Normalization constant
    Norm = np.exp(l*R)*np.sqrt(l*(np.power(k,2)+np.power(l,2)))/np.sqrt(np.pi)
    Norm *= 1/np.sqrt((-1+np.exp(2*l*R))*np.power(k,2)-np.power(l,2)+
            np.power(l,2)*np.cos(2*k*R)-k*l*np.sin(2*k*R))

    #Value of derivative of phi at r
    val = np.cos(k*r)*np.exp(-1*l*r)*k*(np.power(r,-1))
    val -= np.sin(k*r)*np.exp(-1*l*r)*(np.power(r,-2))
    val -= np.sin(k*r)*np.exp(-1*l*r)*l*(np.power(r,-1))

    #return normalized derivative
    return val *Norm

#2nd derivative of wavefunction
def d2psi(r,par):
    #parameters of model
    k = np.pi/R
    l = par

    #Normalization constant
    Norm = np.exp(l*R)*np.sqrt(l*(np.power(k,2)+np.power(l,2)))/np.sqrt(np.pi)
    Norm *= 1/np.sqrt((-1+np.exp(2*l*R))*np.power(k,2)-np.power(l,2)+
            np.power(l,2)*np.cos(2*k*R)-k*l*np.sin(2*k*R))

    #Value of 2nd derivative at r
    val  = 2.0*np.sin(k*r)*(np.power(r,-3))
    val += np.power(r,-2)*(2*l*np.sin(k*r)-2*k*np.cos(k*r))
    val += np.power(r,-1)*((l**2)*np.sin(k*r)-2*k*l*np.cos(k*r)-
                            (k**2)*np.sin(k*r))
    val = val*np.exp(-1*l*r)

    #Return normalized value
    return val * Norm

#Magnitude of approximation function (should be 1)
def f1(r,params):
    return np.power(psi(r,params),2)*4*np.pi*np.power(r,2)

#Expectation value of approximation (or test function)
def f2(r,params):
    val = -(d2psi(r,params)*np.power(r,2)+(2*np.power(r,1))*dpsi(r,params))
    val -= 2*np.power(r,1)*psi(r,params)
    val *= psi(r,params)
    val *= 4*np.pi
    return val

#evaluate the energy more precisely
def FuncPrecise(ind):
    par = ind[0]*SearchPar
    n = 10000
    x0 = R/n
    xArr = np.linspace(x0,R,n)
    val1 = 0.0
    val2 = 0.0
    for i in range(n-1):
        val1 = rk4(xArr[i],val1,f1,xArr[i+1]-xArr[i],par)
        val2 = rk4(xArr[i],val2,f2,xArr[i+1]-xArr[i],par)

    return (10000-val2/val1,)
