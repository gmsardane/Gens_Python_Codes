__all__ = ["lnlikedNdW"]

import distdNdW

import os
import numpy as np
from scipy.integrate import trapz
from scipy import integrate

def lnlikedNdW(p, data):
    print "Parameters are ", p
    if (p[0] < 0.1) or (p[0] > 0.3) or (p[1] < 0) or (p[1] > 1.1) or (p[2] < 0.3) or (p[2] > 0.5) or (p[3] < 0.) or (p[3] > 0.5):
       print "loglike = INVALID PARAMETERS" 
       return -np.inf
       
    else:
        Wwk = p[0]
        Nprime = p[1]
        Wstr = p[2]
        Nstr = p[3]
        x = np.array(data[:,0]) #np.array(dataCaII["W"])
        y = np.array(data[:,2]) #np.array(dataCaII["gW"])/np.max(dataCaII["gW"])
        M = len(x)
        args = y * distdNdW.dNdW(x, p)
        xx=np.arange(0.2, 3., 0.1)
        A=trapz(args, x)
        gtot=np.sum(np.log(y))
        dist=distdNdW.dNdW(x, p)
        ftot=np.sum(np.log(dist))
        LL= -M * np.log(A) + gtot + ftot
        print "loglike = ", LL
        return LL
