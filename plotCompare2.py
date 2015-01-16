from numpy import *
import numpy as np
from matplotlib import *
import matplotlib as mp
from matplotlib.mlab import rec2csv
from matplotlib import rc, rcParams
from scipy import stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern']})

#Plotting Aug 2012 version of York's
dtype=[('mjd', 'i'), ('plate', 'i'), ('fiber', 'i'), ('zabs', '<f8'), ('zsys', '<f8'), 
('wtdz', '<f8'), ('EW1', '<f8'), ('EW2', '<f8'), ('ymjd', '<f8'), ('yplate', '<f8'), 
('yfiber', '<f8'), ('newzsys', '<f8'), ('grade', '|S5'), ('Mg_IIa', '<f8'), ('Mg_IIa_err', '<f8'),
('Mg_IIb', '<f8'), ('Mg_IIb_err', '<f8')]
data=np.genfromtxt("CompareYorkv2GenMatchCorr.txt",delimiter=" ", names=True, dtype=dtype)
dtype=[('mjd', 'i'), ('plate', 'i'), ('fiber', 'i'), ('zabs', '<f8'), ('wtdzbad', '<f8'), 
('newzsys', '<f8'), ('Mg_IIa', '<f8'), ('Mg_IIa_err', '<f8'), ('grade', '|S5'), ('X', '<f8'), 
('EW1', '<f8'), ('err', '<f8'), ('EW2', '<f8'), ('err_1', '<f8'),('wtdz', '<f8')]
#File in NewZabsplotCompare001LT3Av2=24673 but only ~24300 are good with automatic
#Combined GoodDoubleGaussforDensity.txt, GoodSingleGaussfr1103AllInfo.txt & summaryNewZabsv24density.txt
data2=np.genfromtxt("2KMissinginYorkcorrZabs.YorkInfo",delimiter=" ", names=True,dtype=dtype)
dtype=[('mjd', 'i'), ('plate', 'i'), ('fiber', 'i'), ('zabs', '<f8'), ('lambda1', '<f8'), 
('sig1', '<f8'), ('EW1', '<f8'), ('lambda2', '<f8'), ('sig2', '<f8'), ('EW2', '<f8'), 
('newzsys', '<f8'), ('Mg_IIa', '<f8'), ('Mg_IIa_err', '<f8'), ('grade', '|S5')]
data3=np.genfromtxt("sBadTOTInteractive.txt", delimiter=" ", names=True,
comments='#',dtype=dtype) #Measure using measurelines
dtype=[('mjd', 'i'), ('plate', 'i'), ('fiber', 'i'), ('zabs', '<f8'), ('wtdz', '<f8'), 
('newzsys', '<f8'), ('Mg_IIa', '<f8'), ('Mg_IIa_err', '<f8'), ('grade', '|S5'), ('EW1', '<f8'), 
('err1', '<f8'), ('EW2', '<f8'), ('err2', '<f8')]
data4=np.genfromtxt("FixOutliersDensity.YorkInfo", delimiter=" ", names=True, dtype=dtype) #Measure using measurelines
dtype=[('num', 'i'), ('mjd', 'i'), ('plate', 'i'), ('fiber', 'i'), ('zabs', '<f8'), 
('lambda1', '<f8'), ('sig1', '<f8'), ('EW1', '<f8'), ('lambda2', '<f8'), ('sig2', '<f8'), 
('EW2', '<f8'), ('newzsys', '<f8'), ('Mg_IIa', '<f8'), ('Mg_IIa_err', '<f8'), ('grade', '|S5')]
data5=np.genfromtxt("sReDoFixOutliersUnres.YorkInfo", delimiter=" ", names=True,dtype=dtype) #Measure using measurelines
#Added stuff from : MeaUnresResfordensity.txt 
mjd=np.concatenate((np.array(data["mjd"]),np.array(data2["mjd"]),np.array(data3["mjd"]),
np.array(data4["mjd"]),np.array(data5["mjd"])))
plate=np.concatenate((np.array(data["plate"]),np.array(data2["plate"]),np.array(data3["plate"]),
np.array(data4["plate"]),np.array(data5["plate"])))
fiber=np.concatenate((np.array(data["fiber"]),np.array(data2["fiber"]),np.array(data3["fiber"]),
np.array(data4["fiber"]),np.array(data5["fiber"])))
grade=np.concatenate((np.array(data["grade"]),np.array(data2["grade"]),np.array(data3["grade"]),
np.array(data4["grade"]),np.array(data5["grade"])))
zpitt   =np.array(data["wtdz"])
zpitt2  =np.array(data2["wtdz"])
zpitt3  =np.array(data3["lambda1"])/2796.352 -1.
zpitt4  =np.array(data4["wtdz"])
zpitt5  =np.array(data5["lambda1"])/2796.352 -1.
zpittTOT=np.concatenate((zpitt,zpitt2,zpitt3,zpitt4, zpitt5))
zabs    =np.array(data["zabs"])
zabs2   =np.array(data2["zabs"])
zabs3   =np.array(data3["zabs"])
zabs4   =np.array(data4["zabs"])
zabs5   =np.array(data5["zabs"])
zabsTOT =np.concatenate((zabs,zabs2,zabs3,zabs4,zabs5))
zyork   =np.array(data["newzsys"])
zyork2  =np.array(data2["newzsys"])
zyork3  =np.array(data3["newzsys"])
zyork4  =np.array(data4["newzsys"])
zyork5  =np.array(data5["newzsys"])
zyorkTOT=np.concatenate((zyork, zyork2,zyork3,zyork4,zyork5))
pW2796  =abs(np.array(data["EW1"]))
pW2796b =abs(np.array(data2["EW1"]))
pW2796c =abs(np.array(data3["EW1"]))
pW2796d =abs(np.array(data4["EW1"]))
pW2796e =abs(np.array(data5["EW1"]))
pW2796TOT=np.concatenate((pW2796, pW2796b, pW2796c,pW2796d,pW2796e))
pW2803=abs(np.array(data["EW2"]))
pW2803b=abs(np.array(data2["EW2"]))
pW2803c=abs(np.array(data3["EW2"]))
pW2803d=abs(np.array(data4["EW2"]))
pW2803e=abs(np.array(data5["EW2"]))
pW2803TOT=np.concatenate((pW2803,pW2803b,pW2803c,pW2803d,pW2803e))
yW2796=np.array(data["Mg_IIa"])
yW2796b=np.array(data2["Mg_IIa"])
yW2796c=np.array(data3["Mg_IIa"])
yW2796d=np.array(data4["Mg_IIa"])
yW2796e=np.array(data5["Mg_IIa"])
yW2796TOT=np.concatenate((yW2796,yW2796b,yW2796c,yW2796d,yW2796e))
yErr=np.array(data ["Mg_IIa_err"])
yErr2=np.array(data2["Mg_IIa_err"])
yErr3=np.array(data3["Mg_IIa_err"])
yErr4=np.array(data4["Mg_IIa_err"])
yErr5=np.array(data5["Mg_IIa_err"])
yErrTOT=np.concatenate((yErr,yErr2, yErr3,yErr4,yErr5))
print "LENGTHs", len(yErr), len(zyorkTOT), len(pW2796TOT), len(yErrTOT)
#Remove Gen's and Yorks bad measurements first before any operation
goodzpitt=np.array(zpittTOT[(yW2796TOT > 0) & (pW2796TOT >0.)])
goodzyork=np.array(zyorkTOT[(yW2796TOT > 0) & (pW2796TOT > 0.)])
goodpEW=np.array(pW2796TOT[(yW2796TOT > 0) & (pW2796TOT > 0.)])

goodyEW=np.array(yW2796TOT[(yW2796TOT > 0) & (pW2796TOT > 0.)])
goodyErr=np.array(yErrTOT[(yW2796TOT > 0) & (pW2796TOT > 0.)])
goodMJD=mjd[(yW2796TOT > 0) & (pW2796TOT > 0.)]
goodplate=plate[(yW2796TOT > 0) & (pW2796TOT > 0.)]
goodfiber=fiber[(yW2796TOT > 0) & (pW2796TOT > 0.)]
goodgrade=grade[(yW2796TOT > 0) & (pW2796TOT > 0.)]
goodzabs =zabsTOT[(yW2796TOT > 0) & (pW2796TOT > 0.)]
#Convert to REWs using wtzabs
pREW=goodpEW/(1.0+goodzpitt)
yREW=goodyEW/(1.0+goodzyork)
yErr=goodyErr
zabs=goodzabs
offsets=3e5*(goodzpitt-goodzyork)
#offsetsd4=3e5*(zpitt4-zyork4)
#offsetsd5=3e5*(zpitt5-zyork5)
Woffsets=np.array(pREW-yREW)/np.array(yREW) #Percent (Relative) Difference from York cut by
#Woffsetsd4=(pW2796d/(1.+zpitt4)-yW2796d/(1.+zyork4))/(yW2796d/(1.+zyork4))
#Woffsetsd5=(pW2796e/(1.+zpitt5)-yW2796e/(1.+zyork5))/(yW2796e/(1.+zyork5))
ratio=pREW/yREW

#x= np.concatenate((pREW[(ratio < 0.50) & (ratio > 2.0) & (np.abs(offsets) <= 400)],pW2796d/(1.+zpitt4), pW2796e/(1.+zpitt5) ))
#y= np.concatenate((yREW[(ratio < 0.50) & (ratio > 2.0) & (np.abs(offsets) <= 400)],yW2796d/(1.+zyork4), yW2796e/(1.+zyork5)))
#z= np.concatenate((offsets[(ratio < 0.50) & (ratio > 2.0) & (np.abs(offsets) <= 400)],offsetsd4,offsetsd5))
#W=np.concatenate((Woffsets[(ratio < 0.50) & (ratio > 2.0) & (np.abs(offsets) <= 400)],Woffsetsd4,Woffsetsd5))
#gooderr=np.concatenate((yErr[(ratio < 0.50) & (ratio > 2.0) & (np.abs(offsets) <= 400)],yErr4,yErr5))

x= pREW[(np.abs(offsets) <= 300) ]
#& (yREW <=2) & (pREW <= 1)]#,pW2796d/(1.+zpitt4), pW2796e/(1.+zpitt5) ))
y= yREW[(np.abs(offsets) <= 300)] 
#(yREW <=2) & (pREW <= 1)]#,yW2796d/(1.+zyork4), yW2796e/(1.+zyork5)))
z= offsets[(np.abs(offsets) <= 300) ]
#(yREW <=2) & (pREW <= 1)]#,offsetsd4,offsetsd5))
W=Woffsets[(np.abs(offsets) <= 300)] 
#(yREW <=2) & (pREW <= 1)]#,Woffsetsd4,Woffsetsd5))
gooderr=yErr[(np.abs(offsets) <= 300)] 
#(yREW <=2) & (pREW <= 1)]#,yErr4,yErr5))


#xup= pREW[(np.abs(offsets) <= 400) & (yREW > 2) | (pREW > 1)]
#yup= yREW[(np.abs(offsets) <= 400) & (yREW > 2) | (pREW > 1)]
#zup= offsets[(np.abs(offsets) <= 400) & (yREW > 2) | (pREW > 1)]
#Wup=Woffsets[(np.abs(offsets) <= 400) & (yREW > 2) | (pREW > 1)]
#gooderrup=yErr[(np.abs(offsets) <= 400) & (yREW > 2) | (pREW > 1)]

#x=np.concatenate((xlow, xup))
#y=np.concatenate((ylow, yup))
#z=np.concatenate((zlow, zup))
#W=np.concatenate((Wlow, Wup))
#gooderr=np.concatenate((gooderrlow, gooderrup))



#Find out outliers only in the plot:
xmjdBAD=goodMJD[(np.abs(offsets) <= 300)]
xpltBAD=goodplate[(np.abs(offsets) <= 300)]
xfbrBAD=goodfiber[(np.abs(offsets) <= 300)]
xgrdBAD=goodgrade[(np.abs(offsets) <= 300)]
xzbsBAD=goodzabs[(np.abs(offsets) <= 300)]
xzptBAD=goodzpitt[(np.abs(offsets) <= 300)]
xzykBAD=goodzyork[(np.abs(offsets) <= 300)]
xpEWBAD=goodpEW[(np.abs(offsets) <= 300)]

xyEWBAD=goodyEW[(np.abs(offsets) <= 300)]
xyErrBAD=goodyErr[(np.abs(offsets) <= 300)]

#Out=np.array(zip(xmjdBAD,xpltBAD,xfbrBAD, xzbsBAD, xzptBAD, xzykBAD,xyEWBAD, xyErrBAD, xpEWBAD))
#np.savetxt('test.txt', Out, fmt='%i %i %i %f %f %f %f %f %f')
Out=np.array(zip(mjd,plate,fiber, zabsTOT, zpittTOT,pW2796TOT))
np.savetxt('test.txt', Out, fmt='%i %i %i %f %f %f')
print np.shape(Out), len(pW2796TOT), len(x), len(xmjdBAD)
#Out2=np.array(xgrdBAD)
#np.savetxt('test2.txt', Out2, fmt='%s')

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
zmin = z.min()
zmax = z.max()
Wmin = W.min()
Wmax = W.max()

#Look at the absolute ratios.
muWoff=W.mean()
medianWoff = np.median(W)
sigmaWoff = W.std()

#The OUTLIERS
#The Zeroes 
badMJD0=mjd[(yW2796TOT == 0)]
badplate0=plate[(yW2796TOT == 0) ]
badfiber0=fiber[(yW2796TOT == 0) ]
badgrade0=grade[(yW2796TOT == 0) ]
badzpitt0=zpittTOT[(yW2796TOT == 0) ]
badzyork0=zyorkTOT[(yW2796TOT == 0) ]
badpWA0=pW2796TOT[(yW2796TOT == 0) ]
badyWA0=yW2796TOT[(yW2796TOT == 0) ]
badpWB0=pW2803TOT[(yW2796TOT == 0) ]
badyErr=yErrTOT[(yW2796TOT == 0) ]
badzabs=zabsTOT[(yW2796TOT == 0) ]

#Out=np.array(zip(badMJD0,badplate0,badfiber0,badzabs,badzpitt0,badzyork0,badyWA0,badyErr,badpWA0,badpWB0))
#np.savetxt('test.txt', Out, fmt='%i %i %i %f %f %f %f %f %f %f')
#Out2=np.array(badgrade0)
#np.savetxt('test2.txt', Out2, fmt='%s')

#The velocity and REW outliers
#vbadMJD  =mjd[(yW2796TOT >= 0) & (np.abs(zpittTOT-zyorkTOT) > 400)]
#vbadplate=plate[(yW2796TOT >= 0) & (np.abs(zpittTOT-zyorkTOT) > 400)]
#vbadfiber=fiber[(yW2796TOT >= 0) & (np.abs(zpittTOT-zyorkTOT) > 400)]
#vbadzpitt=zpittTOT[(yW2796TOT >= 0) & (np.abs(zpittTOT-zyorkTOT) > 400)]
#vbadzyork0=zyorkTOT[(yW2796TOT >= 0) & (np.abs(zpittTOT-zyorkTOT) > 400)]
#vbadpW27960=pW2796TOT[(yW2796TOT >= 0) & (np.abs(zpittTOT-zyorkTOT) > 400)]
#vbadyW27960=yW2796TOT[(yW2796TOT >= 0) & (np.abs(zpittTOT-zyorkTOT) > 400)]
#vbadyErr=yErrTOT[(yW2796TOT >= 0) & (np.abs(zpittTOT-zyorkTOT) > 400)]
#These are the outliers in the plot
#WbadMJD  =goodMJD[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]
#Wbadplate=goodplate[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]
#Wbadfiber=goodfiber[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]
#Wbadgrade=goodgrade[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]
#Wbadzpitt=goodzpitt[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]
#Wbadzyork=goodzyork[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]
#WbadpEW=goodpEW[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]
#WbadyEW=goodyEW[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]
#WbadyErr=goodyErr[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]
#Wbadzabs=zabs[(np.abs(Woffsets) > 0.50) & (np.abs(Woffsets) < 0.70)]

#Output these bad ones into a table of:

#Out=np.array(zip(WbadMJD,Wbadplate,Wbadfiber, Wbadzabs, Wbadzpitt, Wbadzyork,WbadyEW, WbadyErr, WbadpEW ))
#print np.shape(Out), len(pREW), len(x)
#np.savetxt('test.txt', Out, fmt='%i %i %i %f %f %f %f %f %f')
#Out2=np.array(Wbadgrade)
#np.savetxt('test2.txt', Out2, fmt='%s')
#print len(WbadMJD), len(ratio), len(x)

plt.figure()
plt.plot(pREW,yREW,marker="o", alpha=0.65, linestyle="none")
#cm.YlOrRd_r
plt.axis([xmin, ymax, ymin, ymax]) #Put them on the same scale
plt.title("York vs Pitt")
plt.xlabel(r'$W^{\lambda 2796 } [\AA]$  [PITT]', size=17)
plt.ylabel(r'$W^{\lambda 2796 } [\AA]$  [YORK]', size=17)
plt.plot(x, x, color='k', linestyle='-')
#plt.plot(x, 2*x, color='k', linestyle='-')
#plt.plot(x, 0.5*x, color='k', linestyle='-')
print len(x)
#cb = plt.colorbar()
#cb.set_label(r'$log_{10}N$')
#slope=1
#intercept=0
#plt.plot(x, intercept + slope*x, color='k', linestyle='-')
#slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
plt.show()

#print len(badMJD0)
#print len(vbadMJD)
#print len(WbadMJD)




#===============================Density plot of new York vs Pitt===============================
ax=plt.figure()
plt.hexbin(x,y, bins='log', gridsize=100, cmap=cm.jet, alpha=1, mincnt=1)
#cm.YlOrRd_r
plt.axis([xmin, xmax, ymin, xmax]) #Put them on the same scale
#plt.axis('equal') #Put them on the same scale

plt.title("York vs Pitt")
plt.xlabel(r'$W^{\lambda 2796 } [\AA]$  [PITT]', size=17)
plt.ylabel(r'$W^{\lambda 2796 } [\AA]$  [YORK]', size=17)
cb = plt.colorbar()
cb.set_label(r'$log_{10}N$')
slope=1
intercept=0
plt.plot(x, intercept + slope*x, color='k', linestyle='-')
#plt.plot(x, 0.75+x, color='k', linestyle='-')
#plt.plot(x, -0.5+x, color='k', linestyle='-')

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#plt.plot(x, intercept + slope*x, color='red', linestyle='-')

plt.savefig("NewZabs27KREWsv2.pdf",dpi=400)
print "[m,b, std_err]"
print slope, intercept, std_err
print len(x)
print "RMS Error", np.sqrt(np.mean((x - y) ** 2))
#===============================And the distributions of these===============================
plt.figure()
bins=arange(0,6.5,0.1)
plt.hist(x,bins=bins, histtype='step',color='black',label="Pitt")
plt.hist(y,bins=bins, histtype='step',color='red',label="York", alpha=0.5)
plt.ylabel("Number", size=17)
plt.xlabel(r'$W^{\lambda 2796} [\AA]$', size=17)
plt.legend(loc="upper right")
plt.savefig("NewZabs27KREWDistv2.pdf",dpi=400)
#=====================================Now plot the offsets======================================
### 1.0 Redshift
plt.figure()
plt.hexbin(x,z,bins='log', gridsize=100, cmap=cm.jet, alpha=1, mincnt=1)
plt.axis([xmin, xmax, -400, 400])
plt.title("Redshift Offsets York vs Pitt: 27K")
plt.xlabel(r'$W^{\lambda 2796 } [\AA]$  [PITT]', size=17)
plt.ylabel(r'Redshift Offsets kms$^{-1}$: Pitt-York ', size=17)
cb = plt.colorbar()
cb.set_label(r'$log_{10}N$')
plt.savefig("NewZabs27KOffsetsv2.pdf",dpi=400)
print len(z)
#=====================================Dist of the REDSHIFT Offsets==================================
ax=plt.figure()
bins=arange(-300,300,10)
muz = z.mean()
median = np.median(z)
sigma = z.std()
plt.hist(z,bins=bins,histtype='step',color='black')
plt.axis([-400,400, 0, 6000])
plt.ylabel("Number", size=17)
plt.xlabel('Redshift Offsets [kms$^{-1}$]', size=17)
plt.legend(loc="upper left")
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(muz, median, sigma)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.15, 0.85, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig("NewZabs27KOffsetsDist.pdf",dpi=400)
print muz
#===========================================W Relative Offsets======================================#
plt.figure()
plt.hexbin(x, np.array(W*100.), bins='log', gridsize=100, cmap=cm.jet, alpha=1, mincnt=1)
#plt.ylabel(r'$\Delta$ $W^{\lambda 2796} [\AA]$', size=17)
plt.ylabel('% Difference from York', size=17)
plt.xlabel(r'$W^{\lambda 2796 } [\AA]$', size=17)
plt.title("Relative Difference from York of REWs")
print "Wmax-Wmin is ", Wmax, Wmin
plt.axis([xmin, xmax, Wmin*100, Wmax*100])
cb = plt.colorbar()
cb.set_label(r'$log_{10}N$')
plt.savefig("NewZabs27KWPercentDiff.pdf",dpi=400)

bins=arange(-100.0,100,1)
ax=plt.figure()
plt.hist(W*100,bins=bins, histtype='step',color='black')
plt.ylabel("Number", size=17)
plt.xlabel('Woffsets % Relative Difference From York', size=17)
plt.axis([-100,100, 0, 2000])
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(muWoff*100, medianWoff*100, sigmaWoff*100)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.15, 0.85, textstr, fontsize=14, verticalalignment='top', bbox=props)
plt.savefig("NewZabs27KWOffsetsDist.pdf",dpi=400)


