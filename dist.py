__all__ = ["d2NdWdz"]

import numpy as np
import math


#def d2NdWdz(data, p):
## Norm:
#    Nweak =     p[1] * (1.0 + np.array(data[:,0]))**(p[3]) #as a function of z: Here p[1] = Nweak at z=0
## Strong component
#    argstr = (1.0 + np.array(data[:,0]))**(-p[6]) / p[2]
#    #Nstr = (1.0-p[1]) (1.0 + np.array(data[:,0]))**(p[5])
#    term1 = (1.0 - p[1]) * (1.0 + np.array(data[:,0]))**(-p[5]) * np.exp(-data[:,1] * argstr)  * argstr 
#    # Weak component
#    argwk  = (1.0 + np.array(data[:,0]))**(-p[4]) / p[0]
#    term2 =        Nweak * np.exp(-data[:,1] * argwk )  * argwk
#    return np.array(term1 + term2)

#p = [Wwk, Nratio, Wstr, alp_wk, beta_wk, alpha_str, beta_str]

def d2NdWdz(data, p):
	# Strong component
	# Here p[5] is alphawk-bewtawk (beta = p[6])
	term1p1 = (1.0 + np.array(data[:,0]))**(p[6] - p[7])
	argz1   = (1.0 + np.array(data[:,0]))**(-p[7])
	term1p2 =  np.exp(-data[:,1]/p[2] * argz1)   
	term1 = (p[3]) * term1p1 * term1p2/p[2]
	# Weak component
	#Here p[3] is alphastrong-betastrong [betastrong = p[4]]
	term2p1 = (1.0 + np.array(data[:,0]))**(p[4] - p[5])  #weak term
	#fac = np.zeros(len(data))
	#for i in range(len(data)):
	argz2   = (1.0 + np.array(data[:,0]))**(-p[5])
	term2p2 = np.exp(- data[:,1]/p[0] * argz2)
	#    fac[i] = term2p2
	term2 =  p[1] * term2p1 * term2p2 /p[0]#fac/p[0]    
	#fac[i] = t1 * t2
	return (term1 + term2)

#p = [Wwk, Nwk, Wstr, Nstr, alp_wk, beta_wk, alp_str, beta_str]
	
