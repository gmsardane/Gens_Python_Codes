#!/Users/gmsardane/Library/Enthought/Canopy_64bit/User/bin/python
import numpy as np
import numpy.ma as ma

import minuit
import types
import inspect
#import bigfloat

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib.mlab import rec2csv
from matplotlib import rc, rcParams
from scipy import integrate

import math

import scipy as sci
from scipy import stats
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.integrate import trapz


def dNdz(z, gZ,gamma):
	return   gZ*(1.0+z)**(gamma)/np.sqrt(0.3*(1.+z)**3.+0.7) 

def mloglikdNdz(  gamma):

# Input the path
  pathCaII =  np.genfromtxt("pathCaII525RangedR7dR9Total.txt",delimiter=" ", names=True)
  #dataCaII = np.genfromtxt("FullDataCaIIMINUITRANGE.txt",delimiter=" ", names=True)
  dataCaII = np.genfromtxt("FullSensitivityDataRange.txt",delimiter=" ", names=True)
  W = dataCaII["W"]
  zabs = dataCaII["zabs"]
  gzW = dataCaII["gzW"]

#Find g(z,W):
  
#First Case: W>0.3\AA
  #path = pathCaII['Wmin03']
  zgrid =pathCaII['zgrid']
  wh=np.where((W >= 1.0))
  zabs=zabs[wh]
  gzW = gzW[wh]
  
  
  delz1=np.concatenate([zgrid[1:len(zgrid)-1],[zgrid[len(zgrid)-2]]])
  delz=delz1-zgrid[0:len(zgrid)-1]
  delz[len(delz)-1]=delz[len(delz)-2] 
  #path = path[0:len(path)-2]#/delz[0:len(path)-2]
  zgrid=zgrid[0:len(zgrid)-2]
  
  #gZ = np.interp(z,zgrid,path) 

  index = np.argsort(zabs)
  z=zabs[index]
  
  interp_delz = np.interp(z,zgrid,delz[0:len(delz)-1])
  gZ=gzW[index]*interp_delz
  
  
  #zz = np.arange(0.,1.5,0.01)
  #
  A=integrate.trapz(dNdz(z, gZ, gamma),z)
  gtot=np.sum(np.log(gZ))
  ftot= np.sum(np.log((1.0+z)**(gamma)/np.sqrt(0.3*(1.+z)**3.+0.7)))
  LL=-len(zabs)*np.log(A)+gtot+ftot
  
  return -LL
#plt.plot(z, dNdz(z,gZ, 2))
#  plt.show()

m = minuit.Minuit(mloglikdNdz, gamma=0.01, err_gamma=0.0001, limit_gamma=(1.9,3))
m.up=0.5
m.printMode = 1
m.migrad()
m.hesse()
xx=np.arange(0.2,3.0, 0.2)
print m.values
print m.errors