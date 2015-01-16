__all__ = ["lnlike"]

import numpy as np
from scipy.integrate import trapz

import dist
import os
import math

from matplotlib.mlab import rec2csv
from matplotlib import rc, rcParams
from scipy import stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.integrate import simps
from scipy.integrate import trapz

#class lnlike(object):
#def _init_(self):
def lnlike(p, data):
    lnposterior = 0.0
    print "The current parameters are ", p 
    #Wwk
    if (p[0] < 0.1 or p[0] > 0.3):
        print "loglike = INVALID PARAMETERS" 
        return -np.inf
    lnpostosterior = lnposterior - np.log(0.2)
    #Nweak   
    if (p[1] < 0.5 or p[1] > 1.):
        print "loglike = INVALID PARAMETERS" 
        return -np.inf
    lnpostosterior = lnposterior - np.log(1.0)
    #Wstr    
    if (p[2] < 0.3 or p[2] > 1.0): 
        print "loglike = INVALID PARAMETERS" 
        return -np.inf
    lnposterior = lnposterior - np.log(1.2)
    if (p[3] < 0. or p[3] > 1.): 
        print "loglike = INVALID PARAMETERS" 
        return -np.inf
    lnposterior = lnposterior - np.log(1.)
    #alpha weak - betaweak
    if (p[4] < -2.0 or p[4] > 2.0):
        print "loglike = INVALID PARAMETERS" 
        return -np.inf
    lnposterior = lnposterior - np.log(4.0)
    #beta weak
    if (p[5] < -2.0 or p[5] > 2.0):
        print "loglike = INVALID PARAMETERS" 
        return -np.inf
    lnposterior = lnposterior - np.log(4.0)
    #alpha strong - betastrong
    if (p[6] < -2.0 or p[6] > 2.0):
        print "loglike = INVALID PARAMETERS" 
        return -np.inf
    lnposterior = lnposterior - np.log(4.0) 
    if (p[7] < -2.0 or p[7] > 2.0):
        print "loglike = INVALID PARAMETERS"
        return -np.inf
    lnposterior = lnposterior - np.log(4.0) 


    
    
#p = [Wwk 0, Nweak 1, Wstr 2, alp_wk 3, beta_wk 4, alpha_str 5, beta_str 6] 
    # Beta strong (1.0+z)^Beta for Wstar 
#    if (p[6] < -2.0 or p[6] > 2.0):
#        return -np.inf
#    lnposterior = lnposterior - np.log(4.0)

    #print "OK parameters" 
    # Load in the data
    z = data[:,0]
    W0 = data[:,1]
    gzW = data[:,2]
    M = len(z)
    # Load in the SENSITIVITY grid:
    pathfile = os.path.expanduser('~/Documents/Research/CaII_spec/NewspSpec/pathCaII2HALFUnsatDR7DR9.txt')
    sensitivity_list = [l.split() for l in open(pathfile)]
    sensitivity = np.array(sensitivity_list[1:], dtype=float)
    zgrid = sensitivity[:,0]        
    path_mat = np.array([l.split() for l in open("pathMat.txt")], dtype=float)
    shape = np.shape(path_mat)
    model_mat = np.zeros(np.shape(path_mat))
    #model_mat = model_mat[:,0:30]
    Wgrid = np.arange(0.0, 5.1, 0.1)
    for i in range(len(Wgrid)):
		Wi_mat = Wgrid[i] * np.ones(shape[0])
		Wzgrid = np.transpose(np.vstack(( Wi_mat, zgrid)))
		density = dist.d2NdWdz(Wzgrid, p)
		model_mat[:,i] = density
    path_model_mat = model_mat * path_mat 
    WZint = np.sum(0.1 * np.sum(path_model_mat, axis=0))
    
    distribution =  np.array(dist.d2NdWdz(data,p))
    numerator = np.zeros(len(gzW))
    
    for i in range(len(gzW)):
        if (distribution[i]<0):
            temp = 0.0
        if (distribution[i]>0):
           temp = np.array(math.log(distribution[i]*gzW[i] ))
        #print "temp=",i,"i", temp
        numerator[i] =  temp
    numerator=np.sum(numerator)
    
    #gtot=np.sum(np.log(gzW)) 
    #ftot=np.sum(np.log(distribution))
    # This is the loglikelihood.
    LL = numerator - M * np.log(WZint)
    #LL=  gtot + ftot #-M * np.log(WZint) +
    #plt.plot(path_model_mat, 'o')
    #plt.show()
    #model = dist.d2NdWdz(data,p) *gzW
    #plt.plot(W0,model, 'o')
    if np.isposinf(LL) == True:
        print "Log Likelihood is  = NAN"
        return -np.inf
    if np.isnan(LL) == True:
        print "Log Likelihood is  = NAN"
        return -np.inf        
    print "Log Likelihood is  = ", LL
    return lnposterior + LL
    
    
    
#    # Load in the SENSITIVITY grid:
#    pathfile = os.path.expanduser('~/Documents/Research/CaII_spec/NewspSpec/pathCaII2HALFUnsatDR7DR9.txt')
#    sensitivity_list = [l.split() for l in open(pathfile)]
#    sensitivity = np.array(sensitivity_list[1:], dtype=float)
#    zgrid = sensitivity[:,0]
#    delz1 = np.append(zgrid[1:len(zgrid)], zgrid[len(zgrid)-1])
#    delz = delz1 - zgrid
#    delz[len(zgrid) - 1] = delz[len(zgrid) - 2]
#    rows = np.shape(sensitivity)[0]
#    columns = np.shape(sensitivity)[1]
#    delz_mat = np.ones((rows, columns - 1 )) *  delz[:, None]
#    path_mat = sensitivity[:,1: ]/delz_mat 
#    path_mat = path_mat[~np.isnan(path_mat).any(1)]
#    path_mat = path_mat[:,0:51] 
#    # Of shape : (3718, 61) : Wmin = [0.0, 6.0] 
#    zgrid = zgrid[0:len(path_mat)]
#    # Normalize with respect to both z & W
#    # Create a matrix to multiply with path_mat and then sum:
#    model_mat = np.zeros(np.shape(path_mat))
#    Wgrid = np.arange(0.0, 5.1, 0.1)
#    for i in range(len(Wgrid)):
#		Wi_mat = Wgrid[i] * np.ones(len(zgrid))
#		Wzgrid = np.transpose(np.vstack(( Wi_mat, zgrid)))
#		density = dist.d2NdWdz(Wzgrid, p)
#		model_mat[:,i] = density
#    path_model_mat = delz_mat[:, 0:51] * path_mat #model_mat * 
#    np.savetxt("pathMat.txt", path_model_mat, fmt='%10.8f')
