import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
def gaussianRef(x, mu, sigma):
    return  1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def gaussianTarg(x, mu, sigma):
    #return  1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    gauss1 =1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    gauss2 = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu+3) ** 2 / (2 * sigma ** 2))
    return  gauss1 + gauss2

def gaussiaRatio(x, mu, sigma):
    return np.exp(mu * x -0.5 * mu**2)
muRef = 0
sigmaTarg = 1
sigmaRef = 7
muTarg = 1
x = np.linspace(-30, 30, 1000)


fig, axs = plt.subplots()
axs.plot(x, gaussianRef(x, muRef, sigmaRef))
axs.plot(x, gaussianTarg(x, muTarg, sigmaTarg))
plt.show(block = True)


## from uniform to first gaussian
seeds = np.random.uniform(0, 1, 1000)
seeds = np.random.normal(loc = muRef, scale= sigmaRef, size = 10000)
#seeds = np.random.normal(loc = muTarg, scale= sigmaTarg, size = 1000)
refCDF = np.cumsum(gaussianRef(x, muRef, sigmaRef)/np.sum( gaussianRef(x, muRef, sigmaRef)))

seedVal = np.interp(seeds, x, refCDF)
seedVal = gaussianRef(seeds, muRef, sigmaRef)/np.sum( gaussianRef(x, muRef, sigmaRef))
fig, axs = plt.subplots()
axs.plot(x, gaussianRef(x, muRef, sigmaRef))
axsTw = axs.twinx()
axsTw.hist(seeds)
plt.show(block = True)
##
normTarg = gaussianTarg(x, muTarg, sigmaTarg) / np.sum(gaussianTarg(x, muTarg, sigmaTarg))
normRef = gaussianRef(x, muRef, sigmaRef) / np.sum(gaussianRef(x, muRef, sigmaRef))
gaussRatio = normTarg /normRef / np.sum(normTarg /normRef)
#gaussRatio =  gaussianTarg(x, muTarg, sigmaTarg)/gaussianRef(x, muRef, sigmaRef)
#refCDF = np.cumsum(gaussRatio )
fig, axs = plt.subplots()
axs.plot(x,normTarg )
axs.plot(x,normRef)
axs.plot(x,gaussRatio)
plt.show(block = True)
##

ratioCDF = np.cumsum(gaussRatio)


fig, axs = plt.subplots()
axs.plot(x,ratioCDF *refCDF)
axs.plot(x,refCDF)
axs.plot(x,ratioCDF)
plt.show(block = True)


##



gFunc = np.cumsum(gaussRatio)

#gFunc = x + gaussRatio
xFunc = x
fig, axs = plt.subplots()
axs.plot(xFunc , gFunc)
plt.show(block = True)

targetX = np.interp(seeds ,gFunc,x)
fig, axs = plt.subplots()
axs.plot(x, gaussianTarg(x, muTarg, sigmaTarg) )
axs.plot(x, gaussianRef(x, muRef, sigmaRef) )
axsTw = axs.twinx()
axsTw.hist(targetX, alpha = 0.5)
plt.show(block = True)

##
#numpy.interp(x, xp, fp)
#refU = np.interp(seedVal, gFunc,xFunc)
refU = np.interp(seedVal, normRef,gaussRatio)
targetX = np.interp(refU ,gaussRatio,x)
#targetX =  refU

fig, axs = plt.subplots()
axs.plot(x, gaussianTarg(x, muTarg, sigmaTarg) )
axs.plot(x, gaussianRef(x, muRef, sigmaRef) )
axsTw = axs.twinx()
axsTw.hist(targetX, alpha = 0.5)
plt.show(block = True)


##
refMarg = gaussianRef(x, muRef, sigmaRef)

#refMarg = gaussianTarg(x, muTarg, sigmaTarg)/np.sum(gaussianTarg(x, muTarg, sigmaTarg))
refCDF = np.cumsum(refMarg) * refCDF
refX = np.cumsum(x)
fig, axs = plt.subplots()
axs.plot( x, x + refCDF)
plt.show(block = True)


##
refU = np.interp(seeds,x,refCDF)

fig, axs = plt.subplots()
axs.plot(refCDF)
plt.show(block = True)

fig, axs = plt.subplots()
axs.hist(refU)
plt.show(block = True)
##


ratioMarg = gaussianTarg(x, muTarg, sigmaTarg) / gaussianRef(x, muRef, sigmaRef)
#ratioMarg = gaussianTarg(x, muTarg, sigmaTarg) / gaussianTarg(x, muTarg, sigmaTarg)



ratioCDF = np.cumsum(ratioMarg/np.sum(ratioMarg))

fig, axs = plt.subplots()
axs.plot(x, ratioCDF)
plt.show(block = True)


samples = np.interp(refU,ratioCDF,x)

fig, axs = plt.subplots()
axs.plot(x, gaussianTarg(x, muTarg, sigmaTarg))
axsTw = axs.twinx()
axsTw.hist(samples, bins = 50)
plt.show(block = True)

