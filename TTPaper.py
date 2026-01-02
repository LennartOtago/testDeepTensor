import numpy as np
import scipy as scy
import torch
import deep_tensor as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor
import math#
import time
from AklPresFunctions import *
import pandas as pd

lWdt = 0.5
alpha = 0.75
PgWidthPt = 245
TrueCol = [50/255,220/255, 0/255]#'C2'
binCol = 'C0'
fraction = 1.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# df = pd.read_excel('ExampleOzoneProfiles.xlsx')
df = np.loadtxt('testProf.txt')
h0 = 11
h1 = 20.1
h2 = 32.3
h3 = 47.4
h4 = 51.4
h5 = 71.8
a0 = -6.5
a1 = 0
a2 = 1
a3 = 2.8
a4 = 0
a5 = -2.8
a6 = -2
b0 = 288.15
tempParams = h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0
# print the column names
# print(df.columns)

# get the values for a given column
# press = df['Pressure (hPa)'].values #in hectpascal or millibars
# O3 = df['Ozone (VMR)']
press = df[0, :]
O3 = df[1, :]
O3[O3 < 0] = 0

# til max 84.8520 geopotential height
# geopotential to geometric
# H = Z * R_Earth/(Z + R_Earth)
# H = Z * R_Earth/(Z + R_Earth)
R_Earth = 6356
H = 11
print(H * R_Earth / (R_Earth - H))
Z = 91
R_Earth = 6356
print(Z * R_Earth / (Z + R_Earth))
calc_press = np.zeros((len(press) + 1, 1))
calc_press[0] = 1013.25
calc_temp = np.zeros((len(press), 1))
calc_temp[0] = 288.15
calc_press[1:] = press.reshape((len(press), 1))  # hPa
calc_press = np.copy(press)  # hPa
calc_press[0] = 1013.25
actual_heights = np.zeros(len(press) + 1)
actual_heights = np.zeros(len(press))
for i in range(1, len(calc_press)):
    dx, calc_temp[i - 1] = pressure_to_height(calc_press[i - 1], calc_press[i], actual_heights[i - 1], *tempParams)
    actual_heights[i] = actual_heights[i - 1] + dx

calc_temp[-1] = temp_func(actual_heights[-1], h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0)

minInd = 5
maxInd = 50  # 47#51#54#47
VMR_O3 = O3[minInd:maxInd]
SpecNumLayers = len(VMR_O3)

""" analayse forward map without any real data values"""
heights = actual_heights  # [1:]
SpecNumLayers = len(VMR_O3)
# height_values = heights[minInd:maxInd].reshape((SpecNumLayers,1))
# height_values = np.around(heights[minInd:maxInd][::skipInd].reshape((SpecNumLayers,1)),2)
# temp_values =  np.around(calc_temp[minInd:maxInd][::skipInd],2)

MinH = min(heights[minInd:maxInd])
MaxH = max(heights[minInd:maxInd])
##
height_values = np.linspace(MinH, MaxH, SpecNumLayers).reshape(SpecNumLayers)
VMR_O3 = np.interp(height_values, heights, O3, ).reshape((SpecNumLayers, 1))

new_calc_press = np.zeros(SpecNumLayers)
new_calc_press[0] = calc_press[minInd]
new_calc_temp = np.zeros(SpecNumLayers)
for i in range(1, SpecNumLayers):
    dx = height_values[i - 1] - height_values[i]
    dp, new_calc_temp[i - 1] = height_pressure(height_values[i - 1], dx, new_calc_press[i - 1], *tempParams)
    new_calc_press[i] = new_calc_press[i - 1] - dp

new_calc_temp[-1] = temp_func(height_values[-1], h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0)
height_values = np.around(height_values, 2).reshape((SpecNumLayers, 1))
temp_values = np.around(new_calc_temp, 2).reshape((SpecNumLayers, 1))
pressure_values = new_calc_press.reshape((SpecNumLayers, 1))

SpecNumLayers = len(height_values)

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))

ax1.plot(VMR_O3, height_values[:, 0], marker='o', markerfacecolor=TrueCol, color=TrueCol, label=r'true $\bm{x}$',
         zorder=0, linewidth=3, markersize=15)

plt.show(block=True)











R_Earth = 6356#6371 # earth radiusin km
ObsHeight = 500 # in km

MaxAng = np.arcsin((height_values[-1]+ R_Earth) / (R_Earth + ObsHeight))
#MaxAng = [np.arcsin((55+ R_Earth) / (R_Earth + ObsHeight))]

MinAng = np.arcsin((height_values[0] + R_Earth) / (R_Earth + ObsHeight))


##
pointAcc = 0.00085
meas_angChosen = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))[:30]

print(meas_angChosen)


A_lin_dx, tang_heights_lin, extraHeight = gen_forward_map(meas_angChosen,height_values[:,0],ObsHeight,R_Earth)
#A_lin_dx, tang_heights_lin, extraHeight = gen_forward_map(meas_ang2,height_values,ObsHeight,R_Earth)
SpecNumMeas = len(tang_heights_lin)
m = SpecNumMeas

A_lin = gen_sing_map(A_lin_dx, tang_heights_lin, height_values)

tot_r = np.zeros((SpecNumMeas,1))
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] =  np.sqrt( ( height_values[-1] + R_Earth)**2 - (tang_heights_lin[j] +R_Earth )**2)
print('Distance through layers check: ' + str(np.allclose( sum(A_lin_dx.T,0), tot_r[:,0])))

files = '634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects

my_data = pd.read_csv(files, header=None)
data_set = my_data.values

size = data_set.shape
wvnmbr = np.zeros((size[0],1))
S = np.zeros((size[0],1))
F = np.zeros((size[0],1))
g_air = np.zeros((size[0],1))
g_self = np.zeros((size[0],1))
E = np.zeros((size[0],1))
n_air = np.zeros((size[0],1))
g_doub_prime= np.zeros((size[0],1))
g_prime= np.zeros((size[0],1))

for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    F[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])
    g_doub_prime[i] = float(lines[0][148:153])
    g_prime[i] = float(lines[0][155:160])


#load constants in si annd convert to cgs units by multiplying
h = scy.constants.h #* 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b = constants.Boltzmann #* 1e7#in J K^-1
#T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1
R = constants.gas_constant


mol_M = 48 #g/mol for Ozone
#ind = 293
ind = 623
#pick wavenumber in cm^-1
v_0 = wvnmbr[ind][0]#*1e2
#wavelength
lamba = 1/v_0
f_0 = c_cgs*v_0
print("Frequency " + str(np.around(v_0*c_cgs/1e9,2)) + " in GHz")
AParam = ind, wvnmbr, g_doub_prime, g_prime, E, S
AO3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, *AParam)
A = 2*AO3
Ax =np.matmul(A, VMR_O3 * theta_scale_O3)
linNoiseFreeDat = Ax

SNR = 150

np.random.seed(123)

y, gam0  = add_noise_Blokk(Ax,SNR)

org_height_values = np.copy(height_values)
org_temp_values = np.copy(temp_values)
org_pressure_values = np.copy(pressure_values)
org_VMR_O3 = np.copy(VMR_O3)
nonLinA = calcNonLin(tang_heights_lin, A_lin_dx, height_values, pressure_values, ind, temp_values, VMR_O3, wvnmbr, S, E,g_doub_prime,g_prime)
OrgData = np.matmul(AO3 * nonLinA,VMR_O3 * theta_scale_O3)

noise = np.random.normal(0, np.sqrt(1 / gam0), size = OrgData.shape)
nonLinY = (OrgData + noise).reshape((SpecNumMeas,1))
fraction = 1.5
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))

ax1.scatter(nonLinY, tang_heights_lin)
ax1.plot(nonLinY, tang_heights_lin, label = 'noisy')
ax1.plot(Ax, tang_heights_lin, label = 'noise free')
plt.legend()
ax1.set_xscale('log')

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))

ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)


plt.show(block = True)

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))

ax1.plot(pressure_values, height_values)


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))

ax1.plot(temp_values, height_values)

plt.show(block = True)

## new forwrad model
startInd = 23
EndInd = len(org_height_values[startInd::2]) + startInd
print(EndInd)
height_values[startInd:EndInd] = np.copy(org_height_values[startInd::2])
height_values = np.copy(height_values[:EndInd])

temp_values[startInd:EndInd] = np.copy(org_temp_values[startInd::2])
temp_values = np.copy(temp_values[:EndInd])

pressure_values[startInd:EndInd] = np.copy(org_pressure_values[startInd::2])
pressure_values = np.copy(pressure_values[:EndInd])
VMR_O3 = np.copy(org_VMR_O3)
VMR_O3[startInd:EndInd] = np.copy(org_VMR_O3[startInd::2])
VMR_O3 = np.copy(VMR_O3[:EndInd])
print(VMR_O3.shape)
fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))

#ax1.plot(org_VMR_O3,org_height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)
ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)


plt.show(block = True)



SpecNumLayers = len(height_values)
n =len(height_values)

upL = torch.triu(-1*torch.ones((len(VMR_O3),len(VMR_O3))),1) - torch.triu(-1*torch.ones((len(VMR_O3),len(VMR_O3))),2)
lowL =  torch.tril(-1*torch.ones((len(VMR_O3),len(VMR_O3))),-1) - torch.tril(-1*torch.ones((len(VMR_O3),len(VMR_O3))),-2)
diagL = torch.diag(2*torch.ones(len(VMR_O3)))

L = diagL + lowL + upL

#lowC_L = scy.linalg.cholesky(L, lower = True)

##
A_lin_dx, tang_heights_lin, extraHeight = gen_forward_map(meas_angChosen,height_values[:,0],ObsHeight,R_Earth)

A_lin = gen_sing_map(A_lin_dx, tang_heights_lin, height_values)
AO3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, *AParam )
A = 2*AO3
ATA = np.matmul(A.T,A)
y = np.copy(nonLinY)
ATy = np.matmul(A.T, y)
linDat = A @ VMR_O3
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))

ax1.plot(A @ VMR_O3, tang_heights_lin)
ax1.plot(y, tang_heights_lin)
plt.show(block = True)







# Grid
gridSize = 30#150#15
factor = 3.75#1.5
dim = 18
popt, pcov = scy.optimize.curve_fit(pressFunc, height_values[:,0], pressure_values[:,0], p0=[1.5e-1, pressure_values[0,0]])
#print(popt)

GamBounds = [0.8e15, 1.2e16]
LambBounds = [1e-5, 8e-4]


means = torch.zeros(dim-2)
sigmas = torch.zeros(dim-2)
means[3] = popt[1]
means[2] = 288.15
means[1] = 11
means[0] = popt[0]
means[4] = 32.3
means[5] = 0
means[6] = 20.1

means[7] = -6.5

means[8] = 1
means[9] = 2.8
means[10] = 47.4
means[11] = 0
means[12] = 51.4
means[13] = -2.8
means[14] = 71.8
means[15] = -2



h0 = 11
h1 = 20.1
h2 = 32.3
h3 = 47.4
h4 = 51.4
h5 = 71.8
a0 = -6.5
a1 = 0
a2 = 1
a3 = 2.8
a4 = 0
a5 = -2.8
a6 = -2
b0 = 288.15

#means[12] = popt[0]


#means[14] = popt[2]
sigmaP =  20#15#5*2
#sigmaH = 0.4
#sigmaGrad1 = 0.001#0.005
sigmaGrad2 = 0.0001*30#35#0.01 #* 5
sigmaGrad2 = 0.0001*100
sigmas[3] = sigmaP
sigmas[2] = 10#2.2#4.5#*3# * 50
sigmas[1] = 1.5#0.4#* 1.5
sigmas[0] = sigmaGrad2
sigmas[4] = 2.5#1 #* 0.1
sigmas[5] = 0.1#0.05 #0
sigmas[6] = 1*0.7#2 #* 0.1

sigmas[7] = 0.01
sigmas[8] = 0.01 #* 30#0

sigmas[9] = 0.1
sigmas[10] = 2.5 * 0.2
sigmas[11] = 0.1#0.05#0
sigmas[12] = 2.5 * 0.2
sigmas[13] = 0.1 #* 9
sigmas[14] = 6 * 0.5
sigmas[15] = 0.01 #* 50






univarGridFull = [torch.linspace(*GamBounds , gridSize),
                #torch.linspace(LambBounds[0],0.0025, gridSize),
                torch.linspace(5e10,8e12, gridSize),
                torch.linspace(means[0] - sigmas[0] * 1.2*factor, means[0] + sigmas[0] * factor, gridSize), #1.5 * 0.775
                torch.linspace(means[1] - sigmas[1]  * factor, means[1] + sigmas[1]  * factor, gridSize),
                torch.linspace(means[2] - sigmas[2] * 1.1*factor, means[2] + sigmas[2] * factor, gridSize),
                torch.linspace(means[3] - sigmas[3] *factor, means[3] + sigmas[3]* factor, gridSize),
                torch.linspace(means[4] - sigmas[4]  * factor, means[4] + sigmas[4] * factor, gridSize),
                torch.linspace(means[5] - sigmas[5] * factor, means[5] + sigmas[5] * factor, gridSize),
                torch.linspace(means[6] - sigmas[6] *1.1* factor, means[6] + sigmas[6] *factor, gridSize),
                torch.linspace(means[7] - sigmas[7] * factor, means[7] + sigmas[7] * factor, gridSize),
                torch.linspace(means[8] - sigmas[8] * factor, means[8] + sigmas[8] * factor, gridSize),
                torch.linspace(means[9] - sigmas[9] * factor, means[9] + sigmas[9] * factor, gridSize),
                torch.linspace(means[10] - sigmas[10] * factor, means[10] + sigmas[10] * factor, gridSize),
                torch.linspace(means[11] - sigmas[11] * factor, means[11] + sigmas[11] * factor, gridSize),
                torch.linspace(means[12] - sigmas[12] * factor, means[12] + sigmas[12] * factor, gridSize),
                torch.linspace(means[13]- sigmas[13]* factor, means[13]+ sigmas[13] * factor, gridSize),
                torch.linspace(means[14] - sigmas[14] * factor, means[14] + sigmas[14] * factor, gridSize),
                torch.linspace(means[15]- sigmas[15]  * factor, means[15]+ sigmas[15] * factor, gridSize)]

# draw parameter tensor from grid

randInt = torch.randint(low=0, high = gridSize, size = (100,dim))
randParam = torch.empty(size = randInt.size())
for d in range(0,dim):
    randParam[:,d] = univarGridFull[d][randInt[:,d]]
##
# compute torch_marg_post value

def torch_temp(height, params: Tensor) -> Tensor:
    a = torch.ones(height.shape[0])
    b = torch.ones(height.shape[0])

    b0 = params[4]
    a0 = params[9]
    a1 = params[7]
    a2 = params[10]
    a3 = params[11]
    a4 = params[13]
    a5 = params[15]
    a6 = params[17]
    h0 = params[3]
    h1 = params[8]
    h2 = params[6]
    h3 = params[12]
    h4 = params[14]
    h5 = params[16]

    a[height < h0] = a0
    a[h0 <= height] = a1
    a[h1 <= height] = a2
    a[h2 <= height] = a3
    a[h3 <= height] = a4
    a[h4 <= height] = a5
    a[h5 <= height] = a6
    # a[h6 <= x ] = 0

    b[height < h0] = b0
    b[h0 <= height] = b0 + h0 * a0
    b[h1 <= height] = b0 + (h1 - h0) * a1 + h0 * a0
    b[h2 <= height] = a2 * (h2 - h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h3 <= height] = a3 * (h3 - h2) + a2 * (h2 - h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h4 <= height] = a4 * (h4 - h3) + a3 * (h3 - h2) + a2 * (h2 - h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h5 <= height] = a5 * (h5 - h4) + a4 * (h4 - h3) + a3 * (h3 - h2) + a2 * (h2 - h1) + b0 + (h1 - h0) * a1 + h0 * a0
    # b[h6 <= x ] = a4 * (h6-h5) + a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0

    h = torch.ones(height.shape[0])
    h[height < h0] = 0
    h[h0 <= height] = h0
    h[h1 <= height] = h1
    h[h2 <= height] = h2
    h[h3 <= height] = h3
    h[h4 <= height] = h4
    h[h5 <= height] = h5
    # h[h6 <= x] = h6
    return a * (height - h) + b


def torch_press(x, params: Tensor) -> Tensor:
    b = params[2]
    p0 = params[5]
    return torch.exp(-b * x + torch.log(p0))


def torch_composeAforO3(A_lin, temp, press, ind, wvnmbr, g_doub_prime, g_prime, E, S):
    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]  # in cm^-1

    Q = g_doub_prime[ind, 0] * torch.exp(- HitrConst2 * E[ind, 0] / temp) + g_prime[ind, 0] * torch.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / temp)
    Q_ref = g_doub_prime[ind, 0] * torch.exp(- HitrConst2 * E[ind, 0] / 296) + g_prime[ind, 0] * torch.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / 296)
    LineIntScal = Q_ref / Q * torch.exp(- HitrConst2 * E[ind, 0] / temp) / torch.exp(- HitrConst2 * E[ind, 0] / 296) * (
            1 - torch.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                          1 - torch.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3
    C2 = constants.h * constants.c * v_0 * 1e2 / (constants.Boltzmann * temp)
    # plancks function
    Source = (C1 / (torch.exp(C2) - 1))  # in W m^2/cm^3/sr
    # for number density of air molec / m^3 and 1e2 for pressure values from hPa to Pa
    num_mole = press * 1e2 / (constants.Boltzmann * temp)
    kmTom = 1e3  # for dx integration
    # 1e4 for W cm/cm^2 to W cm/m^2 and S[ind, 0] in cm^2 / molec
    theta_scale = num_mole * 1e4 * S[ind, 0] * kmTom

    A_scal = LineIntScal * Source * theta_scale

    A = A_lin * A_scal.T

    return A, 1


def torch_f(ATy, y, B_inv_A_trans_y):
    return torch.matmul(y.T, y) - torch.matmul(ATy.T, B_inv_A_trans_y)


ind_torch = torch.tensor(ind).clone()
wvnmbr_torch = torch.tensor(wvnmbr).clone()
g_doub_prime_torch = torch.tensor(g_doub_prime).clone()
g_prime_torch = torch.tensor(g_prime).clone()
E_torch = torch.tensor(E).clone()
S_torch = torch.tensor(S).clone()

trueParams = torch.zeros(dim)
trueParams[2:] = means[:].clone()
print(trueParams)
torch_AParam = ind_torch, wvnmbr_torch, g_doub_prime_torch, g_prime_torch, E_torch, S_torch

A_lin_torch = torch.tensor(A_lin).clone().detach()
height = torch.tensor(height_values).clone()

T = torch_temp(height[:, 0], trueParams).reshape((height.shape))

P = torch_press(height[:, 0], trueParams).reshape((height.shape))
print(P.shape)

print(A_lin_torch.shape)
CalcA, theatscale = torch_composeAforO3(A_lin_torch, torch.tensor(temp_values).clone().detach(),
                                        torch.tensor(pressure_values).clone().detach(), *torch_AParam)

torch_temp(height[:, 0], randParam[1, :])

torch_press(height[:, 0], randParam[1, :])
##
fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))

ax1.plot(torch_temp(height[:, 0], randParam[1, :]).detach().numpy(), height_values)

plt.show(block=True)

fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))

ax1.plot(torch_press(height[:, 0], randParam[1, :]).detach().numpy(), height_values)

plt.show(block=True)

# fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))
#
# ax1.plot((2 * CalcA) @ VMR_O3, tang_heights_lin)
# ax1.plot(linDat, tang_heights_lin)
# plt.show(block=True)

##
# neglogliks = torch.zeros(xs.shape[0])
#    for i, x in enumerate(xs):
#        k = prior.transform(x)
#        us = model.solve(k)
#        d = model.observe(us)
#        neglogliks[i] = 0.5 * (d - d_obs).square().sum() / var_error
#    return neglogliks

def torch_marg_post(params: Tensor, h, RealMap, Alin, AParam, L, y, means, sigmas) -> Tensor:
    h1Mean = means[6]
    h1Sigm = sigmas[6]

    h2Mean = means[4]
    h2Sigm = sigmas[4]

    h3Mean = means[10]
    h3Sigm = sigmas[10]

    h4Mean = means[12]
    h4Sigm = sigmas[12]

    h5Mean = means[14]
    h5Sigm = sigmas[14]

    # h6Mean = means[6]
    # h6Sigm = sigmas[6]

    a0Mean = means[7]
    a0Sigm = sigmas[7]

    a1Mean = means[5]
    a1Sigm = sigmas[5]

    a2Mean = means[8]
    a2Sigm = sigmas[8]

    a3Mean = means[9]
    a3Sigm = sigmas[9]

    a4Mean = means[11]
    a4Sigm = sigmas[11]

    a5Mean = means[13]
    a5Sigm = sigmas[13]

    a6Mean = means[15]
    a6Sigm = sigmas[15]

    b0Mean = means[2]
    b0Sigm = sigmas[2]

    h0Mean = means[1]
    h0Sigm = sigmas[1]

    sigmaGrad1 = sigmas[0]
    bmean = means[0]
    sigmaP = sigmas[3]
    pmean = means[3]
    betaD = 1e-35
    betaG = 1e-35
    marg_post = torch.zeros(params.shape[0])
    for i in range(0, params.shape[0]):
        x = params[i, :]
        # lamb = x[1]
        delt = x[1]
        gam = x[0]
        h1 = x[8]
        h2 = x[6]
        h3 = x[12]
        h4 = x[14]
        h5 = x[16]
        a0 = x[9]
        a1 = x[7]
        a2 = x[10]
        a3 = x[11]
        a4 = x[13]
        a5 = x[15]
        a6 = x[17]
        b0 = x[4]
        h0 = x[3]
        b = x[2]
        p0 = x[5]

        lamb = delt / gam
        P = torch_press(h[:, 0], x).reshape((n, 1))
        T = torch_temp(h[:, 0], x).reshape((n, 1))

        CalcA, theatscale = torch_composeAforO3(Alin, T, P, *AParam)
        CurrA = RealMap @ CalcA

        Bp = CurrA.T @ CurrA + lamb * L
        LowTri = torch.linalg.cholesky(Bp)

        G = 2 * torch.log(torch.diag(LowTri)).sum()
        # G = g(CurrA, L, lamb)
        ATy = CurrA.T @ y
        B_inv_A_trans_y = torch.cholesky_solve(ATy, LowTri)
        F = torch_f(ATy, y, B_inv_A_trans_y)
        priors = (- ((h0 - h0Mean) / h0Sigm) ** 2 - ((h1 - h1Mean) / h1Sigm) ** 2 - ((h2 - h2Mean) / h2Sigm) ** 2 - (
                (h3 - h3Mean) / h3Sigm) ** 2 - ((h4 - h4Mean) / h4Sigm) ** 2
                  - ((h5 - h5Mean) / h5Sigm) ** 2 - ((a0 - a0Mean) / a0Sigm) ** 2
                  - ((a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2
                  - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2
                  - ((a6 - a6Mean) / a6Sigm) ** 2 - ((a5 - a5Mean) / a5Sigm) ** 2
                  - ((b0 - b0Mean) / b0Sigm) ** 2
                  - ((pmean - p0) / sigmaP) ** 2 - ((bmean - b) / sigmaGrad1) ** 2)
        gamLamPrior = n / 2 * torch.log(lamb) + (m / 2 + 1) * torch.log(gam) - (betaD * lamb * gam + betaG * gam)
        PrevMarg = - 0.5 * G - 0.5 * gam * F
        marg_post[i] = PrevMarg + 0.5 * priors + gamLamPrior - 400
    # print('----')
    # print(marg_post.shape)
    # print(torch.max((marg_post)))
    # print(torch.min((marg_post)))
    return marg_post


torch_y = torch.tensor(y).clone().detach()
torch_means = means
torch_sigmas = sigmas

neg_log_torch_marg_post = lambda params: -torch_marg_post(params, height, torch.eye(y.shape[0]), 2 * A_lin_torch,
                                                          torch_AParam, L, torch_y, torch_means, torch_sigmas)

randVal = neg_log_torch_marg_post(randParam)


reference = dt.UniformReference() # define reference measure
# here you can choose different reference measure e.g. Gaussian

print(len(univarGridFull))
dim = len(univarGridFull)
bounds = torch.zeros(size = [dim,2])
for i in range(0,dim):
    #currGrid = torch.tensor(np.copy(univarGridFull[i]))
    currGrid = univarGridFull[i]
    bounds[i,0] = currGrid[0]
    bounds[i,1] = currGrid[-1]

approximation_domain = bounds # set grid boundaries
preconditioner = dt.UniformMapping(approximation_domain, reference) # define preconditioner

#dim = 2  # dimension of target density
basis = dt.Lagrange1(num_elems=30) # piecewise linear interpolation
# here you can choose other interpolation basis such as fourier or chebyshev
bases = dt.ApproxBases(basis, dim) # set bases

# may adjust earlier set options such as ranks number of sweep or allow increase of rank by not specifying tt_method="fixed_rank"
tt_options = dt.TTOptions(max_als=1, init_rank=10, tt_method="fixed_rank") # set number of sweeps (max_als=1), ranks, fix ranks
tt = dt.TT(tt_options)
ftt = dt.FTT(bases, tt)
neg_log_torch_marg_post = lambda params: -torch_marg_post(params, height, torch.eye(y.shape[0]), 2*A_lin_torch, torch_AParam , L, torch_y,torch_means, torch_sigmas)

# defined above
target_func = dt.TargetFunc(neg_log_torch_marg_post) # set target function
bridge = dt.SingleLayer()  # set single-layer DIRT (i.e., SIRT)
# do DIRT (layered) as  : dirt = dt.DIRT(target_func, preconditioner, ftt)#, bridge) # do single-layer DIRT (i.e., SIRT)
sirt = dt.DIRT(target_func, preconditioner, ftt ,bridge)

startTime = time.time()
num_sampl = 20_00
# Draw a set of uniform random samples
rs = reference.random(n=num_sampl, d=dim)

# Transform the samples according to SIRT approximation
xs, neglogfxs_sirt = sirt.eval_irt(rs)
# Compute potential function of the (unnormalised) target density at each SIRT sample
neglogfxs_exact = target_func(xs)
res = dt.run_independence_sampler(xs, neglogfxs_sirt, neglogfxs_exact)
print(f'Time to gernerate {num_sampl} samples: {(time.time()-startTime):.2f}s')

print(f"Acceptance rate: {res.acceptance_rate:.3f}")
for i in range(0,dim):
    print(f"IACT (x1): {res.iacts[i]:.3f}")
#print(f"IACT (x2): {res.iacts[1]:.3f}")

# autocorrelation time with Ulli Wollf
from puwr import tauint
import numpy as np
for i in range(0,dim):
    Uwerrmean, Uwerrgam, Uwerrtintgam, Uwerrd_tintgam = tauint([[np.array(res.xs[:,i])]], f =  0)
    print(f'IATC of (x{i}) is: {2* Uwerrtintgam:.2f} \u00B1 {2*Uwerrd_tintgam:.2f}')

fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))

ax1.hist(res.xs[::2, 1], bins=40)

plt.show(block=True)


labels = [r'$\gamma$',r'$\delta$',r'$b$',r'$h_{T,1}$',r'$T_0$',r'$p_0$',r'$h_{T,3}$',r'$a_1$',r'$h_{T,2}$',r'$a_0$',r'$a_2$',r'$a_3$',r'$h_{T,4}$',r'$a_4$',r'$h_{T,5}$',r'$a_5$',r'$h_{T,6}$',r'$a_6$']

alpha = 0.5

fig, axs = plt.subplots(18, 9, figsize = (421/ 72.27, 600/ 72.27), tight_layout = True)
for i in range(0, 18):
    if i <= 8:
        #axs[i, i].plot(univarGridPT[i], margPDFPT[i])
        axs[i, i].hist(res.xs[::2,i], alpha=alpha)
    for j in range(0, 9):
        if j <= 8 and j < i:
            axs[i,j].scatter(res.xs[::2,i],res.xs[::2,j],  s = 0.1, color = 'k' )
            if i < 9:
                axs[j, i].text(0.5, 0.5,f'{scy.stats.pearsonr(res.xs[::2,i],res.xs[::2,j])[0]:.2f}',
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=axs[j, i].transAxes)

        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['bottom'].set_visible(False)
        axs[i, j ].spines['left'].set_visible(False)
        axs[i, j].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False)

for i in range(0, 18):
    axs[i, 0].set_ylabel(str(labels[i]), rotation=0)
    axs[i, 0].yaxis.set_label_coords(-0.2, 0.5)

for j in range(0, 9):
    axs[-1,j].set_xlabel(labels[j])

j = 1
i = 0
for i in range(0, 18):
    for j in range(0, 9):
        if ((i == 1 and j == 0)
                or (i == 0 and j == 1)
        or (i == 2 and j == 1) or (i == 1 and j == 2)
        or (i == 3 and j == 1) or (i == 1 and j == 3)
        or (i == 4 and j == 1) or (i == 1 and j == 4)):
            axs[i, j].spines['top'].set_visible(True)
            axs[i, j].spines['right'].set_visible(True)
            axs[i, j].spines['bottom'].set_visible(True)
            axs[i, j].spines['left'].set_visible(True)
            axs[i, j].spines['bottom'].set_color('r')
            axs[i, j].spines['top'].set_color('r')
            axs[i, j].spines['left'].set_color('r')
            axs[i, j].spines['right'].set_color('r')






plt.show(block = True)


##
fig, axs = plt.subplots(18, 9, figsize = (421/ 72.27, 600/ 72.27), tight_layout = True)


for i in range(0, 18):
    if i >= 9:
        #axs[i, i-8].plot(univarGridPT[i], margPDFPT[i])
        axs[i, i - 9].hist(res.xs[::2,i], alpha = alpha)
    for j in range(0, 18):
        if j >= 9 and j < i:
            axs[i,j-9].scatter(res.xs[::2,i],res.xs[::2,j], s = 0.1, color = 'k' )
            axs[j, i-9].text(0.5, 0.5, f'{scy.stats.pearsonr(res.xs[::2,i],res.xs[::2,j])[0]:.2f}',
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=axs[j, i-9].transAxes)
        if j <= 8 and j < i and i > 8:
            axs[j, i-9].text(0.5, 0.5, f'{scy.stats.pearsonr(res.xs[::2,i],res.xs[::2,j])[0]:.2f}',
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=axs[j, i-9].transAxes)

        #axs[i, j - 8].axis('off')
        axs[i, j - 9].spines['top'].set_visible(False)
        axs[i, j - 9].spines['right'].set_visible(False)
        axs[i, j - 9].spines['bottom'].set_visible(False)
        axs[i, j - 9].spines['left'].set_visible(False)
        axs[i, j - 9].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False)

for i in range(0, 18):
    axs[i, -1].set_ylabel(labels[i],rotation=0)
    axs[i, -1].yaxis.set_label_coords(1.2,0.4)


for j in range(0, 9):
    axs[-1,j].set_xlabel(labels[j+9])



plt.show(block = True)