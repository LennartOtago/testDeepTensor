import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def Target(x, mu, sigma):
    gauss1 =1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    gauss2 = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu+5) ** 2 / (2 * sigma ** 2))
    return  gauss1 + gauss2

def Ratio(x, mu, sigma):
    return np.exp(mu * x -0.5 * mu**2)
muRef = 0
sigmaTarg = 1
sigmaRef = 3
muTarg = 1
x = np.linspace(-10, 7, 100)


fig, axs = plt.subplots()
axs.plot(x,Target(x, muTarg, sigmaTarg))
plt.show(block = True)


## from uniform to first gaussian
seeds = np.random.uniform(0, 1, 1000)

Layers = 3
ratioMarg = Target(x, muTarg, sigmaTarg)**(1/Layers) /np.sum(Target(x, muTarg, sigmaTarg)**(1/Layers))


fig, axs = plt.subplots()
axs.plot(x, ratioMarg)

plt.show(block = True)


ratioCDF = np.cumsum(ratioMarg)
firstX = np.interp(seeds,ratioCDF,x)

fig, axs = plt.subplots()
axs.hist(firstX)
plt.show(block = True)
## layer
currX = np.copy(firstX)
currMarg = np.copy(ratioMarg)

##
for i in range(Layers):
    fig, axs = plt.subplots()
    axs.plot(x, currMarg/np.sum(currMarg))
    axs.plot(x, Target(x, muTarg, sigmaTarg)/np.sum(Target(x, muTarg, sigmaTarg)))
    plt.show(block=True)
    ##
    currCDF = np.cumsum(currMarg/np.sum(currMarg))
    #currCDF = np.cumsum(ratioMarg / np.sum(ratioMarg))
    fig, axs = plt.subplots()
    axs.plot(x, currCDF)
    plt.show(block=True)

    currRefU = np.interp(currX, x, currCDF)
    fig, axs = plt.subplots()
    axs.hist(currRefU)
    plt.show(block=True)

    uptX = np.interp(currRefU,ratioCDF,x)
    fig, axs = plt.subplots()#
    axs.plot(x, currMarg)
    axsTw = axs.twinx()
    axsTw.hist(uptX)
    plt.show(block=True)
    currX = np.copy(uptX)
    currMarg = np.copy(currMarg) * ratioMarg


##

targetMarg = Target(x, muTarg, sigmaTarg)/np.sum(Target(x, muTarg, sigmaTarg))
targetCDF = np.cumsum(targetMarg)
targetX = np.interp(seeds,targetCDF,x)

fig, axs = plt.subplots()
axs.plot(x, Target(x, muTarg, sigmaTarg))
axsTw = axs.twinx()
axsTw.hist(currX)
axsTw.hist(targetX)
plt.show(block = True)

