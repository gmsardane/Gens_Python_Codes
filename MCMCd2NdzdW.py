import numpy as np

import emcee
import math

import plusloglik

# Read in the data file.
data = np.array([l.split() for l in open("FullSensitivityData441.txt")], dtype=float)
# Initial guess.
#p0[0] = Wwk, p0[1] = N, p0[2] = Wstr, p0[3] = alpha, p0[4] = beta
p0 = [ 0.13688926,0.8,0.46625114, 0.05, 0.001 ,0.01, 0.001, 0.001]
ndim = len(p0)
#[ 0.24979568,0.9999475,0.59163943,-1.99327044,1.99907231] #[
def sample():
    # We'll sample with 300 walkers.
    nwalkers = 600
 
    # Generate an initial guess for each walker.
    Wwk0     = np.absolute(p0[0] + 0.001 * np.random.randn(nwalkers))
    Nwk0  = np.absolute(p0[1] + 0.01 * np.random.randn(nwalkers))
    Wstr0    = np.absolute(p0[2] + 0.001 * np.random.randn(nwalkers))
    Nstr0 = np.absolute(p0[3] + 0.001 * np.random.randn(nwalkers))
    alphawk0 = p0[4] + 0.01 * np.random.randn(nwalkers) #alpha - beta  : weak
    betawk0  = p0[5]  + 0.01 * np.random.randn(nwalkers)
    alphastr0 = p0[6] + 0.01 * np.random.randn(nwalkers) #alpha - beta : strong
    betastr0  = p0[7] + 0.01 * np.random.randn(nwalkers) 
	    
    print "Initializing walkers ...."
    initial_mat = np.transpose(np.vstack((Wwk0,     Nwk0,    Wstr0, Nstr0,  alphawk0,    betawk0,  alphastr0,  betastr0)))
    
    initial = initial_mat #.tolist()
    #initial = [p0 + 0.1 * np.random.randn(len(p0)) for k in range(nwalkers)]
    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, plusloglik.lnlike, threads=4,
            args=[data, ] )

    # Do a burn in.
    pos, prob, state = sampler.run_mcmc(initial, 500)

    # Clear and start again.
    sampler.reset()
    sampler.run_mcmc(pos, 2000)
    print "Sampling ... "
    # Take the last 1000 steps from each walker.
    chain = sampler.flatchain
    samples = chain[-1000:, :]
    np.savetxt('chain.txt', samples)
#   np.savetxt('chain.txt', np.columstck samples)

    # Compute the mean and standard deviation values for the slope and
    # intercept given these samples.
    Wwk,      Nwk,     Wstr, Nstr,    alphawk,    betawk ,  alphastr,    betastr       = np.mean(samples, axis=0) #
    WwkErr, NwkErr, WstrErr, NstrErr, alphawkErr, betawkErr , alphastrErr, betastrErr  =  np.std(samples, axis=0) #
    print("These are the mean results from the last parameter chain of 1000 : ")
    print(Wwk,     Nwk,    Wstr,  Nstr, alphawk,    betawk,  alphastr,  betastr)#
    print("And the following are the errors in the parameters : ")
    print(WwkErr, NwkErr, WstrErr, NstrErr, alphawkErr, betawkErr, alphastrErr, betastrErr)#
    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
    print "Done"
    #position = result[0]
    
    

if __name__ == "__main__":
    print "Sampling..."
    sample()
    
    





