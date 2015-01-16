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


def dNdW(W, gW, Wstar):
	return gW * np.exp(-W/Wstar)

dataMgII = np.genfromtxt("FullDataCaIIMINUITRANGE.txt",delimiter=" ", names=True)


#plt.plot(W,dNdW(W,gW,0.5))
#plt.show()
def mloglikMgII(Wstar):
	dataMgII = np.genfromtxt("RatioMore1_8ForMLE.txt",delimiter=" ", names=True)
	x = np.array(dataMgII["W"])
	y = np.array(dataMgII["gW"]) #/np.max(dataMgII["gW"])
	#wh=np.where((x > 0.72))
	#x=x[wh]
	#y=y[wh]
	m = len(x)
	A=integrate.trapz(dNdW(x,y,Wstar),x)
	gtot=np.sum(np.log(y))
	ftot=np.sum(-x/Wstar)
	LL=-m*np.log(A)+gtot+ftot
	#print "Area = ", A
	#print "gtot = ", gtot
	#print "ftot = ", ftot
	return -LL
#Now I want to use minuit to minimize the minus-logLikllihood
m = minuit.Minuit(mloglikMgII, Wstar=0.7, err_Wstar=0.01)
m.up=0.5
m.printMode = 0
m.migrad()
m.hesse()
xx=np.arange(0.2,3.0, 0.2)
a1=1./integrate.trapz(np.exp(-xx/m.values["Wstar"])/m.values["Wstar"],xx)
Nstar=a1*m.values["Wstar"]
print "HERE are the results for the dR4 catalog UNBINNED MLE analysis:"
print "MgII a1 (i.e. COefficient)", a1
print "Nstar : vs Lorenzo's Nstar=1.04", Nstar
print "Wstar = ", m.values["Wstar"]
print m.values, " Values MgII " 
print m.errors, " ERRORS MgII " 
#plt.plot(W,res)
#plt.show()
#To get Nstar/Wstar = overall normalization cofficient for exp(-W/Wstar), I have to integrte:
#1./ epx(-W/Wstar)/Wstar ==> plotting coefficient =~ 1.5
#From this coefficient a1, multiply to get N* =  Nstar*Wstar = 1.04
#i.e work with units consitency!!
###############Now translating this process for the distribution of CaII###################	


#################################Attempting to FIT ALL-together################################

def dNdWCaIIALL(WCaII,gWCaII, Wwk, Nprime, Wstr):
    temp = gWCaII * (Nprime*np.exp(-WCaII/Wstr) + np.exp(-WCaII/Wwk))
    if np.sum(temp) == np.inf:
       return 9999. * np.ones(len(temp))
    else:
       #print "Result: ", gWCaII * (Nprime*np.exp(-WCaII/Wstr) + np.exp(-WCaII/Wwk))
       return gWCaII * (Nprime*np.exp(-WCaII/Wstr) + np.exp(-WCaII/Wwk))

def mloglikCaIIALL(Wwk, Nprime, Wstr): 
	dataCaII = np.genfromtxt("FullDataCaIIMINUITRANGE.txt",delimiter=" ", names=True)
	x = np.array(dataCaII["W"])
	y = np.array(dataCaII["gW"])/np.max(dataCaII["gW"])
	M = len(x)
	A=integrate.trapz(dNdWCaIIALL(x,y, Wwk, Nprime, Wstr),x) 
	gtot=np.sum(np.log(y))
	dist=(Nprime * np.exp(-x/Wstr) + np.exp(-x/Wwk))
	if np.sum(dist) == np.inf:
	   return np.inf
	else:
	   ftot=np.sum(np.log(dist))
	   LL= -M * np.log(A) + gtot + ftot
	   return -LL
#
m = minuit.Minuit(mloglikCaIIALL, Wwk=1.0/7.18512702159,
 Nprime=0.0958554870639, Wstr=1./2.69464165832, err_Wstr=0.01, err_Wwk=0.01, err_Nprime=0.01, 
 limit_Wstr=(0.3,2), limit_Nprime=(0.,10), limit_Wwk=(0.,1) ) 

m.up=0.5
m.strategy=2
m.printMode = 0
m.migrad()
m.hesse()
m.minos()
print m.values
print m.errors
#The error ellipse can be obtained as :
#bands = np.array(m.contour("Wstr", "Wwk", 1, 40))
#plt.figure(0)
#plt.plot(bands[:,0], bands[:,1], linestyle='-')
#plt.xlabel("$W_{strong}^*$", size=17)
#plt.ylabel("$W_{weak}^*$", size=17)
#plt.savefig("ErrorBands.pdf",dpi=400)
xx=np.arange(0.16, 3.5, 0.5)
dist= (np.exp(-xx/m.values["Wwk"])/m.values["Wwk"] + m.values["Nprime"] * 
    np.exp(-xx/m.values["Wstr"])/m.values["Wstr"])
a1=1.0/trapz(dist, xx)
Nwk =  a1
Nstr = (m.values["Nprime"]) * Nwk  
#Calculate error in the weak using the covariances
#The relevant derivatives are :
dWk = -0.16 * math.exp(-0.16/m.values["Wwk"])/ m.values["Wwk"]**2 #  + math.exp(-0.16/m.values["Wwk"])
dWstr = -0.16 * m.values["Nprime"] * math.exp(-0.16/m.values["Wstr"])/ m.values["Wstr"]**2
dNprime = math.exp(-0.16/m.values["Wstr"])

#
ders = np.matrix([dWk, dWstr, dNprime]) #row matrix
covariances = np.matrix([[m.covariance[('Wwk',  'Wwk')] , m.covariance[('Wwk', 'Wstr')], m.covariance[('Wwk', 'Nprime')]],
            [m.covariance[('Wstr', 'Wwk')] , m.covariance[('Wstr', 'Wstr')], m.covariance[('Wstr', 'Nprime')]],
            [m.covariance[('Nprime', 'Wwk')],m.covariance[('Nprime', 'Wstr')], m.covariance[('Nprime', 'Nprime')]]])   #the covariance matrix 3x3
ErrNweak =  np.sqrt(np.array(ders * covariances * ders.transpose())) * (a1**2)
ErrNstr = m.values["Nprime"] * Nwk * np.sqrt( (m.errors["Nprime"]/m.values["Nprime"])**2 + (ErrNweak/Nwk)**2)
#
##print "Fit in Nweak = " , Nwk
#print "Fit in Nstr = " , Nstr
#print "Error in Nweak = " , ErrNweak
#print "Error in Nstr  = " , ErrNstr
#
print "Fit of ALL 441 systems"
print  "[",Nwk, ", -", 1./m.values['Wwk'], ",", Nstr, ", -", 1./m.values['Wstr'], "]"
print "Error of the fits"
print  "[",ErrNweak, ",", m.errors['Wwk']/m.values['Wwk']**2, ",", ErrNstr, ",", m.errors['Wstr']/m.values['Wstr']**2, "]"


#fx = []
#fy = []
#for i in bands:
#	fx.append(i[0])
#	fy.append(i[1])
#fx.append(bands[0][0])
#fy.append(bands[0][1])

#plt.plot(fx,fy, linewidth=1)
#plt.show()

#==========Here I want to examine the redshift evolution of the dNdW profiles==============
#Redshift bin 1:

dataCaII = np.genfromtxt("FullDataCaIIMINUITRANGE.txt",delimiter=" ", names=True)
x = np.array(dataCaII["W"])
y = np.array(dataCaII["gW"])
z = np.array(dataCaII["zabs"])

#Wbin1 = x[bin1]
#gWbin1 =y[bin1]
weak = np.array([1./7.18534930341, 1./9.06663025, 1./5.66345595])
strong= np.array([1./2.68406362076,0.3, 1./2.])
prime = np.array([0.02162629360693, 0.0232215, 0.0232215])

#[5.78465737329, - 11.8399186461, 0.0467149502874, -2.68406362076]

bins = []
bins.append(np.where( (z >= 0.03) & (z <= 0.41) ))
#bins.append(np.where( (z >= 0.41) & (z < 0.72) ))
#bins.append(np.where( (z >= 0.72) & (z <= 1.34) ))
#bins.append(np.where( (z >= 0.75) & (z <= 1.34) ))

def mloglikCaIIBin(Wwk, Nprime, Wstr):  
	M = len(Wbin1)
	func = dNdWCaIIALL(Wbin1,gWbin1, Wwk, Nprime, Wstr)
	func =  ma.masked_invalid(func)
	A=trapz(func,Wbin1) 
	if (A <= 0.):
	   return np.inf
	if (A >  0.):
	    gtot=np.sum(np.ma.log(gWbin1))
	    dist=(Nprime * np.exp(-Wbin1/Wstr) + np.exp(-Wbin1/Wwk))
	    ftot=np.sum(np.ma.log(dist))
	    LL= -M * np.log(A) + gtot + ftot
	    return -LL
	
	
#for i in bins:
wh=np.where( (z >= 0.60) &(z <= 0.78) )
Wbin10 = x[wh]
gWbin10 = y[wh]
sindex=np.argsort(Wbin10)

Wbin1 = Wbin10[sindex]
gWbin1= gWbin10[sindex]
#print Wbin1
#print gWbin1

#j = bins.index(i)
print "minimum rest equivalent width is : ", np.amin(Wbin1)
Wmin =  np.amin(Wbin1)
mbin = minuit.Minuit(mloglikCaIIBin, Wwk=weak[0], Nprime=prime[0], Wstr=0.3, limit_Wstr=(0.1,0.4),#, # err_Nprime=0.0001, , err_Wwk=0.001)#, 
               limit_Nprime=(0.,100), limit_Wwk=(0.1,1.0), err_Wstr=0.001 ) #limit_Wstr=(0.2,1.0), 
mbin.up=0.5
mbin.strategy=2
mbin.tol = 1.e-8
mbin.printMode = 1
mbin.migrad()
mbin.hesse()
mbin.minos()
#mbin.minos("Wwk",1)
#mbin.minos("Wstr",1)

xx=np.arange(0.3, 3.0, 0.5)
dist= (np.exp(-xx/mbin.values["Wwk"])/mbin.values["Wwk"] + mbin.values["Nprime"] * 
 	np.exp(-xx/mbin.values["Wstr"])/mbin.values["Wstr"])
a1=1.0/trapz(dist, xx)
Nwk =  a1#*mbin.values["Wwk"]
Nstr = (mbin.values["Nprime"]) * Nwk #* mbin.values["Wstr"]
print "Nweak = ", Nwk 
dWk = -Wmin * math.exp(-Wmin/mbin.values["Wwk"])/ mbin.values["Wwk"]**2 #  + math.exp(-0.16/m.values["Wwk"])
dWstr = -Wmin * mbin.values["Nprime"] * math.exp(-Wmin/mbin.values["Wstr"])/ mbin.values["Wstr"]**2
dNprime = math.exp(-Wmin/mbin.values["Wstr"])


ders = np.matrix([dWk, dWstr, dNprime]) #row matrix
covariances = np.matrix([[mbin.covariance[('Wwk',  'Wwk')] , mbin.covariance[('Wwk', 'Wstr')], mbin.covariance[('Wwk', 'Nprime')]],
         [mbin.covariance[('Wstr', 'Wwk')] , mbin.covariance[('Wstr', 'Wstr')], mbin.covariance[('Wstr', 'Nprime')]],
         [mbin.covariance[('Nprime', 'Wwk')],mbin.covariance[('Nprime', 'Wstr')], mbin.covariance[('Nprime', 'Nprime')]]])   #the covariance matrix 3x3
ErrNweak =  np.sqrt(np.array(ders * covariances * ders.transpose())) * (a1**2)
ErrNstr = mbin.values["Nprime"] * Nwk * np.sqrt( (mbin.errors["Nprime"]/mbin.values["Nprime"])**2 + (ErrNweak/Nwk)**2)

print mbin.values
print mbin.errors
#	print "Nstrong Bin1 = ", Nstr
#	print "Nweak   Bin1 = ", Nwk
#	print "Inv Wstrong Bin1 = ", 1./m.values['Wstr']
#	print "Inv Wweak   Bin1 = ", 1./m.values['Wwk']
print "j = ", j
print  "[",Nwk, ", -", mbin.values['Wwk'], ",", Nstr, ", -", mbin.values['Wstr'], "]"
print "Error of the fits"
print  "[",ErrNweak, ",", mbin.errors['Wwk'], ",", ErrNstr, ",", mbin.errors['Wstr'], "]"
#mbin.fixed["Wwk"]=True
#mbin.fixed["Wstr"]=True
