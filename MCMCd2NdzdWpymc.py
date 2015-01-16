import numpy as np

import pymc as mc
import math

import os

# Read in the data file.
data = np.array([l.split() for l in open("sFullSensitivityData476.txt")], dtype=float)


# Set up the priors for the parameters
Wwk     = mc.Uniform('Wwk', lower=0.1, upper=0.4)
Nratio  = mc.Uniform('Nratio', lower=0, upper=10.0)
Wstr =  mc.Uniform('Wstr', lower=0.3, upper=1.0)
alphawk = mc.Uniform('alphawk', lower=0., upper=1.)
betawk = mc.Uniform('betawk', lower=0., upper=1.)
alphastr = mc.Uniform('alphastr', lower=0., upper=1.)
betastr  = mc.Uniform('betastr', lower=0., upper=1.)
p = [Wwk.value, Nratio.value, Wstr.value, alphawk.value, betawk.value, alphastr.value, betastr.value]
print p
# Initial guess.
#p0[0] = Wwk, p0[1] = N, p0[2] = Wstr, p0[3] = alpha, p0[4] = beta
p0 = [ 0.1,  0.1 ,  0.4 ,  0.2 ,  0.6,  0.1,  0.1 ]
ndim = 7


@mc.deterministic(trace = False)
def densFunc(value=data, Wwk=Wwk, Nratio=Nratio, Wstr=Wstr, alphawk=alphawk, betawk=betawk, alphastr=alphastr, betastr=betastr):
	term1 = np.exp(-data[:,1]/Wstr * (1.0 +  np.array(data[:,0]) )**(-betastr)) * ((1.0 + np.array(data[:,0]))**(alphastr - betastr)) 
	# Weak component 
	fac = np.zeros(len(data))
	for i in range(len(data)):
	    t1 = np.array(math.exp(- data[i,1]/Wwk))
	    t2 = np.array((1+np.array(data[i,0]))**(-betawk))
	    fac[i] = t1 * t2
	    term2 = Nratio * ((1.0 + np.array(data[:,0]))**(alphawk - betawk)) * fac
	return (term1 + term2)





#print Wwk.value, Nratio.value, Wstr.value, alphawk.value, betawk.value, alphastr.value, betastr.value
# Set up the likelihood
@mc.deterministic(trace = False)
def custom_stochastic(value=data, Wwk=Wwk, Nratio=Nratio, Wstr=Wstr, alphawk=alphawk, alphastr=alphastr, betastr=betastr):
	z = value[:,0]
	W0 = value[:,1]
	gzW = value[:,2]
	M = len(z)
	
	
	
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
	path_mat = path_mat[:,0:61] 
	# Of shape : (3718, 61) : Wmin = [0.0, 6.0] 
	zgrid = zgrid[0:len(path_mat)]
	# Normalize with respect to both z & W
	model_mat = np.zeros(np.shape(path_mat))
	Wgrid = np.arange(0.0, 6.1, 0.1)
	for ii in range(len(Wgrid)):
	    Wi_mat = Wgrid[ii] * np.ones(len(zgrid))
	    Wzgrid = np.transpose(np.vstack(( Wi_mat, zgrid)))
	    dens = densFunc(data=Wzgrid, Wwk=Wwk, Nratio=Nratio, Wstr=Wstr, alphawk=alphawk, betawk=betawk, alphastr=alphastr, betastr=betastr)
	    print dens
	    model_mat[:,ii] = (dens)
	path_model_mat = model_mat * delz_mat[:, 0:61] * path_mat
	WZint = np.sum(0.1 * np.sum(path_model_mat, axis=0))
	gtot=np.sum(np.log(gzW))
	print p
	distribution =  densFunc(data,Wwk.value, Nratio.value, Wstr.value, alphawk.value, betawk.value, alphastr.value, betastr.value)
	ftot=np.sum(np.log(distribution))
	# This is the minus loglikelihood.
	LL= -M * np.log(WZint) + gtot + ftot
	return  LL


model = mc.MCMC([Wwk, Nratio, Wstr, alphawk, alphastr, betastr, custom_stochastic])
model.sample(iter=10)
#
print "!"
print(model.stats()['Wwk']['Wwk'])
print(model.stats()['Wstr']['Wstr'])
print(model.stats()['Nratio']['Nratio'])
print(model.stats()['alphawk']['alphawk'])
print(model.stats()['betawk']['betawk'])
print(model.stats()['alphastr']['alphastr'])
print(model.stats()['betastr']['betastr'])

print 'Wwk',mean(model.trace('Wwk')[:])
print 'Wstr',mean(model.trace('Wstr')[:])







