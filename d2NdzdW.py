#!/Users/gmsardane/Library/Enthought/Canopy_64bit/User/bin/python
import numpy as np
import matplotlib as mp
import scipy as sci
import minuit
import types
import inspect
from matplotlib.mlab import rec2csv
from matplotlib import rc, rcParams
from scipy import stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.integrate import trapz
import os
import dist

#Define the form of the distribution. 
#Note that dNdW = Nwk/Wwk * exp(-W/Wwk) + Nstr/Wstr * exp(-W/Wstr) 
#If now, I let Nwk and Wwk vary with z:
#d2N/dWdz =  Nwk/Wwk  * (1.+ z) ^ (alpha-beta) * exp((-W/Wwk)*(1+z)^(-beta)) + 
#           Nstr/Wstr * exp(-(W/Wstr))

#First define the distribution for dNdzdW as


#def dNdzdW(Wwk, N, Wstr, alpha, beta, gzW, z, WCaII):
#    term1 = np.exp(-WCaII/Wstr)
#    term2 = N * ((1.0 + z)**(alpha + beta)) * (np.exp(-(WCaII/Wwk) * (1+z)**(beta) ))
#    return gzW * (term1 + term2)
#

#Define the combined likelihood as a function of the parameters because this is what
#pyMINUIT wants
def mloglike(Wwk, Nratio, Wstr, gammawk, betawk, gammastr, betastr):
	# Load in the Measurements
	data_list = [l.split() for l in open("sFullSensitivityData476.txt")]
	data = np.array(data_list[1:], dtype=float)
	z = data[:,0]
	W0 = data[:,1]
	gzW = data[:,2]#/np.max(data[:,2])
	M = len(z)
	
	p = [Wwk, Nratio, Wstr, gammawk, betawk, gammastr, betastr]
	
	# Load in the SENSITIVITY grid:
	pathfile = os.path.expanduser('~/Documents/Research/CaII_spec/NewspSpec/pathCaIIUnsat53sigDR7DR9.txt')
	sensitivity_list = [l.split() for l in open(pathfile)]
	sensitivity = np.array(sensitivity_list[1:], dtype=float)
	
	zgrid = sensitivity[:,0]
	delz1 = np.append(zgrid[1:len(zgrid)], zgrid[len(zgrid)-1])
	delz = delz1 - zgrid
	delz[len(zgrid) - 1] = delz[len(zgrid) - 2]
	rows = np.shape(sensitivity)[0]
	columns = np.shape(sensitivity)[1]
	delz_mat = np.ones((rows, columns - 1 )) *  delz[:, None]
	path_mat = sensitivity[:,1: ]/delz_mat 
	path_mat = path_mat[~np.isnan(path_mat).any(1)]
	path_mat = path_mat[:,0:31] #/np.max(path_mat)
	# Of shape : (3718, 61) : Wmin = [0.0, 6.0] 
	zgrid = zgrid[0:len(path_mat)]
	
	# Normalize with respect to both z & W
	# Create a matrix to multiply with path_mat and then sum:
	model_mat = np.zeros(np.shape(path_mat))
	Wgrid = np.arange(0.0, 3.1, 0.1)
	for i in range(len(Wgrid)):
		Wi_mat = Wgrid[i] * np.ones(len(zgrid))
		Wzgrid = np.transpose(np.vstack(( Wi_mat, zgrid)))
		density = dist.d2NdWdz(Wzgrid, p)
		
		model_mat[:,i] = density
	path_model_mat = model_mat * delz_mat[:, 0:31] * path_mat
	WZint = np.sum(0.1 * np.sum(path_model_mat, axis=0))
	
	gtot=np.sum(np.log(gzW))
	distribution =  dist.d2NdWdz(data,p)
	ftot=np.sum(np.log(distribution))
	
	# This is the loglikelihood.
	LL=  gtot + ftot
	return -LL
	
p0 = [0.1, 0.01, 0.3, 0.4, 0.2]
m = minuit.Minuit(mloglike,  Wwk=0.11, Nratio=0.01, Wstr=0.5, 
gammawk=0.1, betawk=0.1, gammastr=0.1, betastr=0.1, limit_Wwk= (0.1, 0.2), limit_Nratio=(0.0,0.5),
limit_Wstr=(0.3,2.0), limit_gammawk=(-2,2), limit_gammastr=(-2,2),
limit_betawk=(-2.,2), limit_betastr=(-2.,2), err_Wwk=0.001) 
#m.fixed["Wwk"]=True
#alphastr=0.1, betastr=0.1,, limit_betastr=(0., 1)
m.up=0.5
m.strategy=2
m.printMode = 1
m.migrad()
m.hesse()
m.minos()
#alphastr=0.1, betastr=0.1,
print(m.values)
print(m.errors)


    




#
#Wwk=0.19
#N=2
#Wstr=0.3
#alpha=1.2 
#beta=1.5
#
#print(mloglik(Wwk, N, Wstr, alpha, beta))
#
	
#Wwk= 0.100003  
#N=2.14109     
#Wstr=0.300001
#alpha=1.2      
#beta=-1.2326

