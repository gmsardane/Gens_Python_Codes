import numpy as np

import emcee
import math

import plusloglikdNdW
import os

#pathfile = os.path.expanduser('~/Documents/Research/CaII_spec/CaIIBlindSearch/FullDataCaIIMINUITRANGE.txt')
pathfile = os.path.expanduser('~/Documents/Research/CaII_spec/MCMCPython/CaIIDRbinDRlt.txt')

#FullDataCaIIMINUITRANGE.tx
data_list = [l.split() for l in open(pathfile)]
data = np.array(data_list[1:], dtype=float)
p0 = [ 0.2,  0.9 ,  0.4 , 0.04 ]
ndim=len(p0)
z=np.array(data[:,0])
gW=np.array(data[:,2])
W=np.array(data[:,1])
wh=np.where((z >= 0.02))#  & (z < 0.41))
zbin0  =  z[wh]
gWbin0 = gW[wh]
Wbin0  =  W[wh]
sindex=np.argsort(Wbin0)##
Wbin = Wbin0[sindex]
gWbin= gWbin0[sindex]
zbin = zbin0[sindex]
databin=(np.transpose(np.vstack((Wbin, zbin, gWbin))))
def sample():
    # We'll sample with 2000 walkers.
    nwalkers = 2000
 
    # Generate an initial guess for each walker.
    Wwk0 =    p0[0] + 0.001 * np.random.rand(nwalkers)
    Nweak0  = p0[1] + 0.001 * np.absolute(np.random.rand(nwalkers))
    Wstr0 =    p0[2] + 0.001 * np.random.rand(nwalkers)
    Nstr0  = p0[3] + 0.001 * np.absolute(np.random.rand(nwalkers))
    
    print "Initializing walkers ...."
    initial = np.transpose(np.vstack((Wwk0, Nweak0,Wstr0, Nstr0)))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, plusloglikdNdW.lnlikedNdW, args=[databin, ])
    # Do a burn in.
    pos, prob, state = sampler.run_mcmc(initial, 1000)
    # Clear and start again.
    sampler.reset()
    sampler.run_mcmc(pos, 2000)
    print "Sampling ... "
    chain = sampler.flatchain
    samples = chain[-1000:, :]
    np.savetxt('chaindNdWBin1b.txt', samples, fmt='%10.8f')
    # intercept given these samples.
    samples = chain[-1000:, :]
    Wwk, Nweak, Wstr, Nstr  = np.mean(samples, axis=0)
    WwkErr, NweakErr, WstrErr, NstrErr  = np.std(samples, axis=0)
    print("These are the mean results from the last parameter chain of 1000 : ")
    print(" Wstar_Weak ", "Nstar_weak", "Wstar_str", "Nstar_str")
    print(Wwk, Nweak, Wstr, Nstr)
    print("And the following are the errors in the parameters : ")
    print(WwkErr, NweakErr, WstrErr, NstrErr)
    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
    print "Done"

if __name__ == "__main__":
    print "Sampling..."
    sample()

#Bin 1:
#These are the mean results from the last parameter chain of 1000 : 
#(0.68468268971195134, 0.9426162896135617, 0.1439508319150791)
#And the following are the errors in the parameters : 
#(0.16041331700265515, 0.032529015251820531, 0.016112992910448445)
#Mean acceptance fraction: 0.360
#Bin2
#(0.45736302252842148, 0.83711224983263977, 0.22260649672600374)
#And the following are the errors in the parameters : 
#(0.18608016026921939, 0.17626695538761086, 0.040784948478709102)
#Mean acceptance fraction: 0.251
#Sampling ... 
#These are the mean results from the last parameter chain of 1000 : 
#(0.28421748796419838, 0.65019734833195131, 0.19593570735042784)
#And the following are the errors in the parameters : 
#(0.13534153603844984, 0.24651916026438475, 0.059377000950166785)
#Mean acceptance fraction: 0.226
#These are the mean results from the last parameter chain of 1000 : 
#(0.31915385375356731, 0.57839495048134426, 0.19435908056825243)
#And the following are the errors in the parameters : 
#(0.10018708664241921, 0.26611520116609721, 0.047707831528532384)
#Mean acceptance fraction: 0.253
#D
#Done
#Bin 3 vs:
#Sampling ... 
#These are the mean results from the last parameter chain of 1000 : 
#(0.43623206358346156, 0.84323034692141507, 0.17583278744436209)
#And the following are the errors in the parameters : 
#(0.2180787551637163, 0.17345135298212891, 0.04615055249363427)
#Mean acceptance fraction: 0.226
#Done
#
#
#These are the mean results from the last parameter chain of 1000 : 
#(0.332, 0.73952699864825466, 0.17193982403149838)
#And the following are the errors in the parameters : 
#(0.102, 0.28382848857553439, 0.032485311535138089)
#Mean acceptance fraction: 0.225

#If I have 4 bins then:
#Bin 1: [0.60,0.78] has
#These are the mean results from the last parameter chain of 1000 : 
#(0.23157601555934995, 0.39985622502604418, 0.31272374335213515)
#And the following are the errors in the parameters : 
#(0.040605088803448743, 0.31132596999964557, 0.19415525706576389)
#Mean acceptance fraction: 0.260
#Done


#A FOUR PARAMETER FIT RESULTS IN THE FF: YEHEY!! THEY ARE CONSISTENT WITH THE THREE PARAM FIT
#Sampling ... 
#These are the mean results from the last parameter chain of 1000 : 
#(0.14921548617574307, 0.79398524669203008, 0.65914787475707093, 0.045665639021654589)
#And the following are the errors in the parameters : 
#(0.018394238938749401, 0.27026904195489404, 0.16212840332360109, 0.024843315959300942)
#Mean acceptance fraction: 0.262
# BIN 2
#These are the mean results from the last parameter chain of 1000 : 
#(' Wstar_Weak ', 'Nstar_weak', 'Wstar_str', 'Nstar_str')
#(0.20908362004312953, 0.61542956717376007, 0.31160953292433757, 0.74897087462276124)
#And the following are the errors in the parameters : 
#(0.044316832643197683, 0.27807876092314726, 0.068817880354102579, 0.43754485405708887)
#Mean acceptance fraction: 0.308
#OR
#These are the mean results from the last parameter chain of 1000 : 
#(' Wstar_Weak ', 'Nstar_weak', 'Wstar_str', 'Nstar_str')
#(0.20255380810981244, 0.77439891173132869, 0.3956805423816841, 0.14777263818865344)
#And the following are the errors in the parameters : 
#(0.035399142611344143, 0.21247639716429151, 0.057680988698836248, 0.080048137543031458)
#Mean acceptance fraction: 0.461
##
#These are the mean results from the last parameter chain of 1000 : 
#(' Wstar_Weak ', 'Nstar_weak', 'Wstar_str', 'Nstar_str')
#(0.16898830802954418, 0.59133538976703603, 0.36282737944070248, 0.10872770339902406)
#And the following are the errors in the parameters : 
#(0.03488431871673827, 0.24575548676810877, 0.062341275435885996, 0.080495565310656672)
#Mean acceptance fraction: 0.452