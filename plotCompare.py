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

#==============================Old Stuff from Dan Nestor============================================

data=np.genfromtxt("Compareztol_001.txt",delimiter=" ", names=True)

wW2796=np.array(data["wW2796"])
yW2796=np.array(data["wMg_IIa"])
z_sys =np.array(data["z_sys"])

x = wW2796[wW2796>0]
y = yW2796[wW2796>0]/(1+z_sys[wW2796>0])
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

plt.figure(0)
plt.hexbin(x,y,bins='log', gridsize=100, cmap=cm.jet, alpha=1)
plt.axis([xmin, ymax, ymin, ymax])
plt.title("York vs Pitt")
plt.xlabel(r'$W^{\lambda 2796 \AA} [\AA]$  [PITT]', size=17)
plt.ylabel(r'$W^{\lambda 2796 \AA} [\AA]$  [YORK]', size=17)
cb = plt.colorbar()
cb.set_label(r'$log_{10}N$')
plt.plot(x, intercept + slope*x, color='k', linestyle='-')
plt.savefig("line2.pdf",dpi=400)

#================================================NewFig=============================================
data=np.genfromtxt("summaryNewZabsv24density.txt",delimiter=" ", names=True, 
dtype=[('file_', '|S23'), ('z_sys','<f8'), ('Mg_IIa','<f8'), ('Mg_IIa_err','<f8'),
('zabs','<f8'), ('wtzabs','<f8'), ('lambda1','<f8'), ('EW1','<f8'),
('lambda2','<f8'), ('EW', '<f8')]) #zero york removed
zpitt =data["wtzabs"]
zabs  =data["zabs"]
zyork =data["z_sys"]
pW2796=data["EW1"]
yW2796=data["Mg_IIa"]
yErr=data["Mg_IIa_err"]
spec=np.array(data["file_"])

pREW = np.array(-pW2796)/np.array(1.0+np.array(zpitt))
yREW = np.array(yW2796)/np.array(1.0 + np.array(zyork))
yErr=np.array(yErr)/np.array(1.0 + np.array(zyork))
offsets=np.array(3e5*np.array(zpitt-zyork))
Woffsets=np.array(pREW-yREW)/np.array(yREW) #Percent (Relative) Difference from York cut by

goodpREW= np.array(pREW[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) <= 0.35)])
goodyREW= np.array(yREW[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) <= 0.35)])
goodoffs= np.array(offsets[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) <= 0.35)])
gooderr=np.array(yErr[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) <= 0.35)])
goodWoffsets=np.array(Woffsets[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) <= 0.35)])
goodratio= np.array(goodpREW/goodyREW)

bname=str(spec[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) > 0.35)])
#print bname[1]
bzyor=zyork[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) > 0.35)]
bWyor=yW2796[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) > 0.35)]
bzpit=zpitt[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) > 0.35)]
bzabs=zabs[(np.abs(offsets) < 400) & (yREW > 0) & (pREW > 0) & (np.abs(Woffsets) > 0.35)]

#Adding about 674 from refit with single gauss
dtype=[('fileSG', '|S23'), ('wtdzSG', '<f8'), ('WLASG', '<f8'), ('sigASG', '<f8'), 
('EW1SG', '<f8'), ('WLBSG', '<f8'), ('sigBSG', '<f8'), ('EW2SG', '<f8'), ('zsysSG', '<f8'), 
('zabs', '<f8'), ('MgIIaSG', '<f8'), ('MgIIaerrSG', '<f8'), ('Grade', '|S5')]
dataSG=np.genfromtxt("GoodSingleGaussfr1103AllInfo.txt",delimiter=" ", names=True, dtype=dtype)

wtdzSG=np.array(dataSG["wtdzSG"])
zsysSG=np.array(dataSG["zsysSG"])
REWASGpitt=np.abs(np.array(dataSG["EW1SG"])/(1.+np.array(dataSG["wtdzSG"])))
REWASGyork=np.array(dataSG["MgIIaSG"])/(1.+np.array(dataSG["zsysSG"]))
offsSG=np.array(3e5*(wtdzSG-zsysSG))
WoffsSG=np.array(REWASGpitt-REWASGyork)/np.array(REWASGyork)

#Adding about 218 DG
dtype=[('fileDG', '|S23'), ('zsysDG', '<f8'), ('Mg_IIaDG', '<f8'), ('Mg_IIa_errDG', '<f8'), 
('zabsDG', '<f8'), ('wtdzDG', '<f8'), ('EW1DG', '<f8'), ('WLADG', '<f8'), ('EW2wrong', '<f8'), 
('lambda2wrong', '<f8'), ('gradeDG', '|S5')]
dataDG=np.genfromtxt("GoodDoubleGaussforDensity.txt",delimiter=" ", names=True, dtype=dtype)

wtdzDG=np.array(dataDG["wtdzDG"])
zsysDG=np.array(dataDG["zsysDG"])
REWADGpitt=np.abs(np.array(dataDG["EW1DG"])/(1.0+np.array(dataDG["wtdzDG"])))
REWADGyork=np.array(dataDG["Mg_IIaDG"])/(1.0+np.array(dataDG["zsysDG"]))
offsDG=np.array(3e5*(wtdzDG-zsysDG))
WoffsDG=np.array(REWADGpitt-REWADGyork)/np.array(REWADGyork)

#Adding those that were measured manually
dtype=[('num', 'i'),('spec', '|S23'),('Mzsys', '<f8') , ('MlambdaA', '<f8'), ('MsigA', '<f8'), 
('MEWA', '<f8'), ('MlambdaB', '<f8'), ('MsigB', '<f8'), ('MEWB', '<f8'), ('YMgIIa' , '<f8') ,
('YMgIIaErr', '<f8'), ('Mgrade', '|S5'), ('Mwtzab', '<f8') ]
dataMan=np.genfromtxt("MeaUnresResfordensity.txt",delimiter=" ", comments='#', names=True)
#Mn stands for manual
wtdzMn=np.array(dataMan["Mwtzab"])
zsysMn=np.array(dataMan["Mzsys"])
REWAMnpitt=np.abs(np.array(dataMan["MEWA"])/(1.0+np.array(dataMan["Mwtzab"])))
REWAMnyork=np.array(dataMan["MYMgIIa"])/(1.0+np.array(dataMan["Mzsys"]))
offsMn=np.array(3e5*(wtdzMn-zsysMn))
WoffsMn=np.array(REWAMnpitt-REWAMnyork)/np.array(REWAMnyork)

goffsMn= offsMn[(abs(offsMn) < 100) & (REWAMnyork > 0.0)]
gwtdzMn=wtdzMn[(abs(offsMn) < 100) & (REWAMnyork > 0.0)]
gzsysMn=zsysMn[(abs(offsMn) < 100) & (REWAMnyork > 0.0)]
gREWAMnpitt=REWAMnpitt[(abs(offsMn) < 100) & (REWAMnyork > 0.0)]
gREWAMnyork=REWAMnyork[(abs(offsMn) < 100) & (REWAMnyork > 0.0)]
gWoffsMn= WoffsMn[(abs(offsMn) < 100) & (REWAMnyork > 0.0)]


#dt=np.dtype([('bname', '|S23'), ('bzyor', '<f8'),('bWyor', '<f8'), ('bzpit', '<f8'), ('bzabs', '<f8')])
#tuplelist=[(bname[1],'bname'), (bzyor[1], 'bzyor'), (bWyor[1],'bWyor'), (bzpit[1],'bzpit'), (bzabs[1], 'bzabs')]
#print tuplelist
#arr=np.array(tuplelist, dtype=dt)
#print(arr['bname'])
#Out=np.array(zip(bname,bzyor,bWyor,bzpit,bzabs),dtype=[('bname', '|S1'), ('bzyor', float),('bWyor', float),
#('bzpit', float), ('bzabs', float)])
#print Out
#np.savetxt('Bad500.txt', Out, fmt=('%s %f %f %f %f'))

#x=np.array(goodpREW[(np.abs(goodratio) < 1.5) & (np.abs(goodratio) > 1.0/1.5)])
#y=np.array(goodyREW[(np.abs(goodratio) < 1.5) & (np.abs(goodratio) > 1.0/1.5)])
#z=np.array(goodoffs[(np.abs(goodratio) < 1.5) & (np.abs(goodratio) > 1.0/1.5)])

x=np.concatenate((goodpREW,REWASGpitt,REWADGpitt, gREWAMnpitt)) #Remeasured pitt REWs
y=np.concatenate((goodyREW,REWASGpitt,REWADGyork, gREWAMnyork)) #Corresponding York Matches
z=np.concatenate((goodoffs,offsSG, offsDG, goffsMn)) #offsets in redshifts
W=np.concatenate((goodWoffsets,WoffsSG,WoffsDG, gWoffsMn)) #offsets in REWs

print "Lengths", len(x), len(W), len(Woffsets)-181

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
zmin = z.min()
zmax = z.max()
Wmax=W.max()
Wmin=W.min()

#York Error Dist: Passed ALL cuts
plt.figure()
bins=arange(0,0.3,0.01)
err=np.array(gooderr)/np.array(goodyREW)
plt.hist(err, bins=bins, histtype='step')
print "ERROR in YORK Range :", err.min(), err.max()
plt.ylabel("Number", size=17)
plt.xlabel(r'York Error Distribution', size=17)
plt.savefig("ErrorDistYork.pdf",dpi=400)

#Density plot of new York vs Pitt
plt.figure()
plt.hexbin(x,y, bins='log', gridsize=100, cmap=cm.jet, alpha=1)
#cm.YlOrRd_r
plt.axis([xmin, ymax, ymin, ymax]) #Put them on the same scale
plt.title("York vs Pitt")
plt.xlabel(r'$W^{\lambda 2796 } [\AA]$  [PITT]', size=17)
plt.ylabel(r'$W^{\lambda 2796 } [\AA]$  [YORK]', size=17)
cb = plt.colorbar()
cb.set_label(r'$log_{10}N$')
slope=1
intercept=0
plt.plot(x, intercept + slope*x, color='k', linestyle='-')
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#plt.plot(x, intercept + slope*x, color='red', linestyle='-')
plt.savefig("NewZabs25KREWs.pdf",dpi=400)
print "[m,b, std_err]"
print slope, intercept, std_err
#And the distributions of these
plt.figure()
bins=arange(0,6.5,0.1)
plt.hist(x,bins=bins, histtype='step',color='black',label="Pitt")
plt.hist(y,bins=bins, histtype='step',color='red',label="York", alpha=0.5)
plt.ylabel("Number", size=17)
plt.xlabel(r'$W^{\lambda 2796} [\AA]$', size=17)
plt.legend(loc="upper right")
plt.savefig("NewZabs25KREWDist.pdf",dpi=400)

###=======================Now plot the offsets...
### 1.0 Redshift
plt.figure()
plt.hexbin(x,z,bins='log', gridsize=100, cmap=cm.jet, alpha=1)
plt.axis([xmin, xmax, -400, 400])
plt.title("Redshift Offsets York vs Pitt: 25K")
plt.xlabel(r'$W^{\lambda 2796 } [\AA]$  [PITT]', size=17)
plt.ylabel(r'Redshift Offsets kms$^{-1}$: Pitt-York ', size=17)
cb = plt.colorbar()
cb.set_label(r'$log_{10}N$')
plt.savefig("NewZabs25KOffsets.pdf",dpi=400)
print len(z)

#Dist of the Absolute Offsets
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
plt.savefig("NewZabs25KOffsetsDist.pdf",dpi=400)
print muz


#Look at the absolute ratios.
WRatio=np.array(x/y)
Rmax=WRatio.max()
Rmin=WRatio.min()
muWoff=W.mean()
medianWoff = np.median(W)
sigmaWoff = W.std()
plt.figure()

bad=Woffsets[np.abs(Woffsets) > 0.30] #Those that differ more than 15% than york? But I should look at the tyical error of york
print "We have this much outside 30% relative from York ", len(bad) 

bins=arange(-1.0,1.0,0.01)
plt.figure()
plt.hexbin(x, WRatio, bins='log', gridsize=200, cmap=cm.jet, alpha=1)
#plt.ylabel(r'$\Delta$ $W^{\lambda 2796} [\AA]$', size=17)
plt.ylabel('Ratio Density Plot: Pitt/York', size=17)
plt.xlabel(r'$W^{\lambda 2796 } [\AA]$', size=17)
print "Wmax-Wmin is ", Wmax, Wmin
plt.axis([xmin, xmax, 0, 2])
cb = plt.colorbar()
cb.set_label(r'$log_{10}N$')
plt.savefig("NewZabs25KWRatio.pdf",dpi=400)

plt.figure()
plt.hexbin(x, np.array(W*100.), bins='log', gridsize=100, cmap=cm.jet, alpha=1)
#plt.ylabel(r'$\Delta$ $W^{\lambda 2796} [\AA]$', size=17)
plt.ylabel('% Difference from York', size=17)
plt.xlabel(r'$W^{\lambda 2796 } [\AA]$', size=17)
plt.title("Relative Difference from York of REWs")
print "Wmax-Wmin is ", Wmax, Wmin
plt.axis([xmin, xmax, Wmin*100, 200])
cb = plt.colorbar()
cb.set_label(r'$log_{10}N$')
plt.savefig("NewZabs25KWPercentDiff.pdf",dpi=400)

#Dist of Offsets
bins=arange(-100.0,100,1)
ax=plt.figure()
plt.hist(Woffsets*100,bins=bins, histtype='step',color='black')
plt.ylabel("Number", size=17)
plt.xlabel('Woffsets % Relative Difference From York', size=17)
plt.axis([-100,100, 0, 2000])
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(muWoff*100, medianWoff*100, sigmaWoff*100)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.15, 0.85, textstr, fontsize=14, verticalalignment='top', bbox=props)
plt.savefig("NewZabs25KWOffsetsDist.pdf",dpi=400)






#plt.subplots_adjust(hspace=0.5)
#plt.subplot(121)
#plt.hexbin(x,y, cmap=cm.jet)
#plt.axis([xmin, xmax, ymin, ymax])
#plt.title("Hexagon binning")
#cb = plt.colorbar()
#cb.set_label('counts')