#!/usr/bin/env python
#------------------------------------------------------------
# Script which demonstrates how to find the best-fit
# parameters of a Voigt line-shape model
# 
# Vog, 26 Mar 2012
#------------------------------------------------------------
import numpy
from matplotlib.pyplot import figure, show, rc
from scipy.special import wofz
from kapteyn import kmpfit
import scipy.signal as signal
ln2 = numpy.log(2)

def voigt(x, y):
   # The Voigt function is also the real part of 
   # w(z) = exp(-z^2) erfc(iz), the complex probability function,
   # which is also known as the Faddeeva function. Scipy has 
   # implemented this function under the name wofz()
   z = x + 1j*y
   I = wofz(z).real
   return I

def Voigt(nu, alphaD, alphaL, nu_0, A):
   # The Voigt line shape in terms of its physical parameters
   f = numpy.sqrt(ln2)
   x = (nu-nu_0)/alphaD * f
   y = alphaL/alphaD * f
   #backg = a + b*nu 
   V = A*f/(alphaD*numpy.sqrt(numpy.pi)) * voigt(x, y)
   return V  #Equivalent to Churchill's: Chap 6.5 u(x,y)

def funcV(p, x):
    # Compose the Voigt line-shape
    alphaD, alphaL, nu_0, I = p
    return Voigt(x, alphaD, alphaL, nu_0, I)  # I =  Area under curve related to the amplitude


#Now I want to plot the profile of a 2e20 - Line in GALEX 


A = -2  
alphaD = 0.5  #Doppler width
alphaL = 0.5  #Lorentzian width

nu_0 = 1216    #Central wavelength
p0 = [alphaD, alphaL, nu_0, A ]

x = numpy.linspace(1150, 1300, 200)
fosc = 0.4164           #Oscillator strength for the Lya resonance absorption 
Lya = 1215.7            #Transition rest wavelength
gamma=0.62508e9         #Damping Constant for Lya in units of per sec            
N = 2e20
xx = (x-nu_0)/alphaD * numpy.sqrt(ln2)
yy = alphaL/alphaD * numpy.sqrt(ln2)
tau = 1.498E-15 * N * fosc * numpy.sqrt(numpy.pi) * Lya * voigt(xx,yy)
#voigt()#funcV(p0, x)
profile = numpy.exp(-tau)


# Plot the result
rc('legend', fontsize=6)
fig = figure()
frame1 = fig.add_subplot(1,1,1)
frame1.plot(x, profile, 'bo', label="data")
label = "Model with Voigt function"
#frame1.plot(x, tau, 'g', label=label)
#frame1.plot((nu_0-hwhm,nu_0+hwhm), (offset+amp/2,offset+amp/2), 'r', label='fwhm')
frame1.set_xlabel("$\\lambda$")
frame1.set_ylabel("$\\phi(\\lambda)$")
title = "Profile data with Voigt"
frame1.set_title(title, y=1.05)
frame1.grid(True)
leg = frame1.legend(loc=3)
show()