import numpy as np
import scipy as scy
from scipy import constants
import time



'''generate forward map accoring to trapezoidal rule'''
def gen_sing_map(dxs, tang_heights, heights):
    m,n = dxs.shape
    A_lin = np.zeros((m,n+1))
    for i in range(0,m):
        t = 0
        while heights[t] <= tang_heights[i]:
            t += 1
        A_lin[i, t - 1:] = gen_trap_rul(dxs[i, t - 1:])
        # A_lin[i, t-1] = 0.5 * dxs[i, t-1]
        # for j in range(t, n):
        #     A_lin[i,j] = 0.5 * (dxs[i,j-1] + dxs[i,j])
        # A_lin[i, -1] = 0.5 * dxs[i, -1]
    return A_lin
    
def pressure_to_height(p0, pplus, x, h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0):
    R = constants.gas_constant
    R_Earth = 6356#6371  # earth radiusin km6356#
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    #temp = get_temp(x)
    temp = temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0)
    dP = pplus - p0
    #return - np.log(p0/pplus) /(-28.97 * grav / R /temp )
    #return ( np.log(pplus) -np.log(p0))/ (-28.97 * grav / R / temp)
    M = 28.97
    return (dP/p0) /(-M * grav / R /temp ), temp

def add_noise_Blokk(Ax,SNR):
    stdNoise = max(Ax)/SNR
    return Ax + np.random.normal(0,stdNoise , (len(Ax), 1)) , 1/stdNoise**2

    
def height_pressure(x, dx, p0,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0):
    R = constants.gas_constant
    R_Earth = 6356#6371  # earth radiusin km6356#
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    temp = temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0)
    #dP = pplus - p0
    M = 28.97
    return dx * (-M * grav / R /temp ) * p0, temp
    
    ''' generate dx'''
def gen_forward_map(meas_ang, heights, obs_height, R):
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    A_height = np.zeros((num_meas, len(heights)-1))

    for m in range(0, num_meas):
        t = 0
        #find t so that layers[t] is larger than tang height
        while heights[t] < tang_height[m]:
            t += 1

        for i in range(t, len(heights)):
            A_height[m, i-1] = np.sqrt((heights[i] + R) ** 2 - (tang_height[m] + R) ** 2) - np.sum( A_height[m, :i])

    return A_height, tang_height, heights[-1]
    
    
''' generate dx'''
def gen_forward_map(meas_ang, heights, obs_height, R):
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    A_height = np.zeros((num_meas, len(heights)-1))

    for m in range(0, num_meas):
        t = 0
        #find t so that layers[t] is larger than tang height
        while heights[t] < tang_height[m]:
            t += 1

        for i in range(t, len(heights)):
            A_height[m, i-1] = np.sqrt((heights[i] + R) ** 2 - (tang_height[m] + R) ** 2) - np.sum( A_height[m, :i])

    return A_height, tang_height, heights[-1]
    

def composeAforO3(A_lin, temp, press, ind, wvnmbr, g_doub_prime, g_prime, E, S):
    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0] # in cm^-1


    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp) + g_prime[ind, 0] * np.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296) + g_prime[ind, 0] * np.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3
    C2 = constants.h * constants.c * v_0 * 1e2 / (constants.Boltzmann * temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1)) # in W m^2/cm^3/sr
    # for number density of air molec / m^3 and 1e2 for pressure values from hPa to Pa
    num_mole = press * 1e2 / (constants.Boltzmann * temp)
    kmTom = 1e3  # for dx integration
    # 1e4 for W cm/cm^2 to W cm/m^2 and S[ind, 0] in cm^2 / molec
    theta_scale = num_mole * 1e4 * S[ind,0] * kmTom

    A_scal = LineIntScal * Source * theta_scale

    A = A_lin * A_scal.T

    return A, 1


def gen_trap_rul(dxs):
    #val = np.zeros(len(dxs)+1)
    sumMat = np.eye(len(dxs)+1)
    Ones = np.ones((len(dxs)+1,len(dxs)+1))
    sumMat = sumMat + np.triu(Ones,1) - np.triu(Ones,2)
    return 0.5*(dxs @ np.copy(sumMat[:-1,:]))

def calcNonLin(tang_heights, dxs,  height_values, pressure_values, ind, temp_values, VMR_O3, wvnmbr, S, E,g_doub_prime,g_prime):
    '''careful that A_lin is just dx values
    maybe do A_lin_copy = np.copy(A_lin/2)
    A_lin_copy[:,-1] = A_lin_copy[:,-1] * 2
    if A_lin has been generated for linear data'''

    # from : https://hitran.org/docs/definitions-and-units/
    # all calc in CGS
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]  # in cm^-1

    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp_values) + g_prime[ind, 0] * np.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / temp_values)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296) + g_prime[ind, 0] * np.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp_values) / np.exp(
        - HitrConst2 * E[ind, 0] / 296) * (
                          1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp_values)) / (
                          1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))


    num_mole = 1 / constants.Boltzmann
    # 1e-4 cm^2/molec to m^2/molec
    theta = num_mole * VMR_O3 * S[ind,0] * 1e-4
    # 1e2 for pressure hPa to Pa and 1e5 for km to m
    ConcVal = - pressure_values * 1e2 * LineIntScal / temp_values * theta * 1e3

    SpecNumMeas = len(tang_heights)
    SpecNumLayers = len(VMR_O3)

    afterTrans = np.zeros((SpecNumMeas, SpecNumLayers))
    preTrans = np.zeros((SpecNumMeas, SpecNumLayers))
    for i in range(0,SpecNumMeas):
        t = 0
        while height_values[t] <= tang_heights[i]:
            t += 1
        flipDxs = np.flip(dxs[i, t - 1:])
        flipVal = np.flip(ConcVal[t - 1:])
        currDxs = gen_trap_rul(np.append(flipDxs, dxs[i, t - 1]))
        ValPerLayAfter = np.sum(np.append(flipVal , ConcVal[t]) * currDxs)
        afterTrans[i, t - 1] = np.exp(ValPerLayAfter)
        for j in range(t-1, SpecNumLayers-1):
            currDxs = gen_trap_rul(dxs[i,j:])
            ValPerLayPre = np.sum(ConcVal[j:].T  * currDxs)
            preTrans[i,j] = np.exp(ValPerLayPre)

            if j >= t:
                currDxs = gen_trap_rul(np.append(flipDxs, dxs[i, t - 1:j]))
                ValPerLayAfter = np.sum(np.append(flipVal , ConcVal[t:j + 1]) * currDxs)
                afterTrans[i, j] = np.exp(ValPerLayAfter)

        currDxs = gen_trap_rul(np.append(flipDxs, dxs[i, t - 1:]))
        ValPerLayAfter = np.sum(np.append(np.flip(ConcVal[t - 1:]), ConcVal[t:]) * currDxs)
        afterTrans[i, -1] = np.exp(ValPerLayAfter)
        preTrans[i, -1] = 1

    return preTrans + afterTrans




def temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0):
    a = np.ones(x.shape)
    b = np.ones(x.shape)

    a[x < h0] = a0
    a[h0 <= x] = a1
    a[h1 <= x] = a2
    a[h2 <= x] = a3
    a[h3 <= x] = a4
    a[h4 <= x ] = a5
    a[h5 <= x ] = a6
    #a[h6 <= x ] = 0

    b[x < h0] = b0
    b[h0 <= x] = b0 + h0 * a0
    b[h1 <= x] = b0 + (h1 - h0) * a1 + h0 * a0
    b[h2 <= x] = a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h3 <= x ] = a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h4 <= x ] = a4 * (h4 -h3) + a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h5 <= x ] = a5 * (h5 -h4) + a4 * (h4 -h3) + a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    #b[h6 <= x ] = a4 * (h6-h5) + a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0


    h = np.ones(x.shape)
    h[x < h0] = 0
    h[h0 <= x] = h0
    h[h1 <= x] = h1
    h[h2 <= x] = h2
    h[h3 <= x] = h3
    h[h4 <= x] = h4
    h[h5 <= x] = h5
    #h[h6 <= x] = h6
    return a * (x - h) + b

def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T, A) + l * L
    # Bu, Bs, Bvh = np.linalg.svd(B)
    upL = scy.linalg.cholesky(B)
    # return np.sum(np.log(Bs))
    return 2 * np.sum(np.log(np.diag(upL)))

def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[0::, 0].T, y[0::, 0]) - np.matmul(ATy[0::, 0].T, B_inv_A_trans_y)




def SQ_postTP(indices, univarGrid, means, sigmas, A, y, height_values, gam, const):
    n = len(height_values)


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

    #h6Mean = means[6]
    #h6Sigm = sigmas[6]

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

    b0Mean = means[1]
    b0Sigm = sigmas[1]

    h0Mean = means[2]
    h0Sigm = sigmas[2]

    sigmaGrad1 = sigmas[3]
    bmean = means[3]

    #sigmaH = sigmas[13]
    sigmaP = sigmas[0]
    pmean = means[0]


    Values = np.zeros(len(indices))
    for j in range(len(indices)):
        h1 = univarGrid[6][indices[j,6].astype(np.int32)]
        h2 = univarGrid[4][indices[j,4].astype(np.int32)]
        h3 = univarGrid[10][indices[j,10].astype(np.int32)]
        h4 = univarGrid[12][indices[j,12].astype(np.int32)]
        h5 = univarGrid[14][indices[j,14].astype(np.int32)]
        a0 = univarGrid[7][indices[j,7].astype(np.int32)]
        a1 = univarGrid[5][indices[j,5].astype(np.int32)]
        a2 = univarGrid[8][indices[j,8].astype(np.int32)]
        a3 = univarGrid[9][indices[j,9].astype(np.int32)]
        a4 = univarGrid[11][indices[j,11].astype(np.int32)]
        a5 = univarGrid[13][indices[j, 13].astype(np.int32)]
        a6 = univarGrid[15][indices[j, 15].astype(np.int32)]
        b0 = univarGrid[1][indices[j,1].astype(np.int32)]
        h0 = univarGrid[2][indices[j, 2].astype(np.int32)]
        b = univarGrid[3][indices[j,3].astype(np.int32)]
        p0 = univarGrid[0][indices[j,0].astype(np.int32)]
        paramT = [h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0]
        paramP = [b, p0]
        PT = pressFunc(height_values[:, 0], *paramP).reshape((n, 1)) / temp_func(height_values, *paramT).reshape((n, 1))

        postDat = - gam * np.sum((y - A @ PT) ** 2)

        #postDat = 0#- ((popt[1] - b2) / sigmaGrad2) ** 2
        #-((h6 - h6Mean) / h6Sigm) ** 2
#+200
        Values[j] =  (postDat - ((h0 - h0Mean) / h0Sigm) ** 2 -  ((h1 - h1Mean) / h1Sigm) ** 2 - ((h2 - h2Mean) / h2Sigm) ** 2 - (
                       (h3 - h3Mean) / h3Sigm) ** 2 -  ((h4 - h4Mean) / h4Sigm) ** 2
                       - ((h5 - h5Mean) / h5Sigm) ** 2  - ((a0 - a0Mean) / a0Sigm) ** 2
                       - ((a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2
                       - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2
                      - ((a6 - a6Mean) / a6Sigm) ** 2 - ((a5 - a5Mean) / a5Sigm) ** 2
                      - ((b0 - b0Mean) / b0Sigm) ** 2
                       - ((pmean - p0) / sigmaP) ** 2 - ((bmean - b) / sigmaGrad1) ** 2) #- 2 * gam[j] * 1e-10
        #- ((popt[1] - h0P) / sigmaH) ** 2
        #Values[j] = postDat
    #print('---')  # + 30
    #const = - np.max( 0.25 * Values)
    #print(np.max( 0.25 * Values)+const)#+ 30
    #print(np.min(0.25 * Values) + const)  # + 30
    #print((y - A @ PT).shape)
    #print(postDat)
    #print(max(Values-postDat))
    #print(min(Values-postDat))
    CurrVAL =  0.25 * Values + const
    print(max(CurrVAL))
    # if any(np.exp(CurrVAL)**2==0):
    #     print('too small')
    # if any(np.isinf(np.exp(CurrVAL)**2)):
    #     #print(CurrVAL[0.25 * Values + const < -350])
    #     print('too large')

    return  0.25 * Values + const

def pressFuncFullFit(x, b1, b2, h0, p0):
    b = np.ones(len(x))
    b[x<=h0] = b1
    b[x>h0] = b2
    return np.exp(-b * (x -h0)  + np.log(p0))


# def pressFunc(x, b, h0, p0):
#     return np.exp(-b * (x -h0)  + np.log(p0))
def pressFunc(x, b, p0):
    return np.exp(-b * x  + np.log(p0))

def MinLogMargPost(params, A, L, ATy, ATA, y, betaG, betaD):

    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    if lamb < 0  or gam < 0:
        return np.nan

    m,n = A.shape


    Bp = ATA + lamb * L

    LowTri = scy.linalg.cholesky(Bp, lower=True)
    B_inv_A_trans_y = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])
    #B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    G = 2 * np.sum(np.log(np.diag(LowTri)))
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)

def LogMargPost(indices, univarGrid, A, lam0, f_coeff, f_0, g_0, betaD, betaG, L, ATA, ATy, y, delG, const):
    Values = np.zeros(len(indices))

    for j in range(len(indices)):
        gam = univarGrid[0][indices[j,0].astype(np.int32)]
        lamb = univarGrid[1][indices[j,1].astype(np.int32)]

        f_0_1 = f_coeff[0]
        f_0_2 = f_coeff[1]
        f_0_3 = f_coeff[2]
        # g_0_1 = g_coeff[0]
        # g_0_2 = g_coeff[1]
        # g_0_3 = g_coeff[2]


        if lamb < 0  or gam < 0:
            return np.nan


        m,n = A.shape


        # Bp = ATA + lamb * L
        # LowTri = np.linalg.cholesky(Bp)
        # UpTri = LowTri.T
        # B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])
        # G = g(A, L, lamb)
        # F = f(ATy, y, B_inv_A_trans_y)

        delta_lam = lamb - lam0
        F = f_0 + f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3 #+ f_0_4 * delta_lam**4 + f_0_5 * delta_lam**5
        #G = g_0 + g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3 #+ g_0_4 * delta_lam**4 + g_0_5 * delta_lam**5
        G = (np.log(lamb) - np.log(lam0)) * delG + g_0
        # taylorG = g_tayl(lamb - minimum[1], g_0, g_0_1, g_0_2, g_0_3, g_0_5, 0 ,0)
        # taylorG = g(A, L, lamb)
        #taylorG = np.exp(GApprox)

        Bp = ATA + lamb * L
        LowTri = scy.linalg.cholesky(Bp, lower=True)
        B_inv_A_trans_y = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])
        #G = 2 * np.sum(np.log(np.diag(LowTri)))
        F = f(ATy, y, B_inv_A_trans_y)
        Values[j] = const + n/2 * np.log(lamb) + (m/2 + 1) * np.log(gam) - 0.5 * G - 0.5 * gam * F -  ( betaD *  lamb * gam + betaG *gam)

        #Values[j] = - lam0
    print(np.exp(0.5*np.min(Values)))
    print(np.exp(0.5*np.max(Values)))
    return np.exp(0.5*Values)

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = 1#(5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim







def runSQTTMargandThenCond(A, L, ATy, ATA, y, gamma0, vari, theta_scale_O3, univarGridO3, index, const):
    fminFuncEval = 25
    betaG = 1e-35  # 2e-9
    betaD = 1e-35
    gridSize = len(univarGridO3[0])
    MinFunc = lambda params: MinLogMargPost(params, A, L, ATy, ATA, y, betaG, betaD)
    TotalStartTime = time.time()
    minimum = scy.optimize.fmin(MinFunc, [gamma0, 1 / gamma0 * 1 / np.mean(vari) / 15], maxiter=fminFuncEval)
    lam0 = minimum[1]
    print(minimum)
    # prepare for fast calculation with taylor expansion
    B = (ATA + lam0 * L)

    LowTri = scy.linalg.cholesky(B, lower=True)
    B_inv_A_trans_y0 = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])

    B_inv_L = scy.linalg.cho_solve((LowTri, True),L)
    # LowTri = np.linalg.cholesky(B)
    # UpTri = LowTri.T
    # for i in range(len(B)):
    #     B_inv_L[:, i] = lu_solve(LowTri, UpTri, L[:, i])

    B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
    B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)

    f_coeff = np.zeros(3)
    #g_coeff = np.zeros(3)
    f_coeff[0] = np.matmul(np.matmul(ATy[:, 0].T, B_inv_L), B_inv_A_trans_y0)
    f_coeff[1] = 0#-1 * np.matmul(np.matmul(ATy[:, 0].T, B_inv_L_2), B_inv_A_trans_y0)
    f_coeff[2] = 0#1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_3), B_inv_A_trans_y0)

    # g_coeff[0] = np.trace(B_inv_L)
    # g_coeff[1] = -1 / 2 * np.trace(B_inv_L_2)
    # g_coeff[2] = 1 / 6 * np.trace(B_inv_L_3)

    f_0 = f(ATy, y, B_inv_A_trans_y0)
    g_0 = 2 * np.sum(np.log(np.diag(LowTri)))

    delG = (g(A, L, univarGridO3[1][-1]) - g(A, L, univarGridO3[1][0])) / (np.log(univarGridO3[1][-1]) - np.log(univarGridO3[1][0]))
    lamMax = lam0 + 0.25 * lam0
    lamMin = lam0 - 0.25 * lam0

    delG = (g(A, L, lamMax) - g(A, L, lamMin)) / (np.log(lamMax) - np.log(lamMin))
    #univarGridO3[1] = np.linspace(150, lam0 + 2* lam0, gridSize)

    ttFunc = lambda indices: LogMargPost(indices, univarGridO3, A, lam0, f_coeff, f_0, g_0, betaD, betaG, L, ATA, ATy, y, delG, const)

    ##
    initRank =5# 9

    dimMargO3 = 2
    startTime = time.time()
    # Run cross
    # random inital cores
    f0 = tt.rand(gridSize, dimMargO3, r=initRank)
    ttTrain = rect_cross.cross(ttFunc, f0, nswp=1, kickrank=0, rf=0, eps=1e-100)
    elapsedTime = time.time() - startTime
    print(f'time for TT Cross Round ' + str(index)+ f': {elapsedTime}')
    '''
        :param myfun: Vectorized function of d variables (it accepts I x D integer array as an input, and produces I numbers as output)
        :type myfun: function handle
    '''
    print(ttTrain)

    TTCore = [None] * dimMargO3
    maxRank = 1

    # Cores of f must be extracted carefully, since we might have discontinuous ps
    core = np.zeros((ttTrain.core).size, dtype=np.float64)
    ps_my = 0
    for i in range(0, ttTrain.d):
        cri = np.copy(ttTrain.core[range(ttTrain.ps[i] - 1, ttTrain.ps[i + 1] - 1)])
        # print(cri)
        #np.savetxt(index +'ttTraincoreMargO3'+ str(i) + '.txt', cri,header=str(ttTrain.r[i]) + ' ,' + str(ttTrain.n[i]) + ',' + str(ttTrain.r[i + 1]))
        TTCore[i] = cri.reshape((ttTrain.r[i], ttTrain.n[i], ttTrain.r[i + 1]), order='F')
        core[range(ps_my, ps_my + ttTrain.r[i] * ttTrain.n[i] * ttTrain.r[i + 1])] = cri
        ps_my = ps_my + ttTrain.r[i] * ttTrain.n[i] * ttTrain.r[i + 1]
        #np.savetxt(index +'uniVarGridMargO3'+ str(i) + '.txt', univarGridO3[i])
        if int(ttTrain.r[i + 1]) > maxRank:
            maxRank = int(ttTrain.r[i + 1])

    ## do marginalisation
    absError = 1#e-5
    #print('absolute Error: ' + str(absError))
    margPDFO3 = getMargfromSQTT(TTCore, univarGridO3, absError)

    ## calculate conditional mean and variance

    IDiag = np.eye(len(B))

    interval = 1#4

    lambMarg = margPDFO3[1][::interval] / np.sum(margPDFO3[1][::interval])
    gamMarg = margPDFO3[0][::interval] / np.sum(margPDFO3[0][::interval])

    lambs = univarGridO3[1][::interval]
    gams = univarGridO3[0][::interval]
    #print(len(gams))
    CondResults = np.zeros((len(gams), len(L)))

    VarB = np.zeros((len(gams), len(L), len(L)))
    gamInt = np.zeros(len(gams))
    meanGamInt = np.zeros(len(gams))
    for i in range(0, len(gams)):

        currB = ATA + lambs[i] * L
        LowTri = scy.linalg.cholesky(currB, lower=True)
        B_inv_A_trans_y = scy.linalg.cho_solve((LowTri, True),  ATy[:, 0])
        #np.allclose(currB @ ( B_inv_A_trans_y ),ATy[0::, 0])
        CondResults[i, :] = B_inv_A_trans_y * lambMarg[i]

        B_inv = scy.linalg.cho_solve((LowTri, True), IDiag)
        # B_inv = np.zeros(currB.shape)
        # # startTime = time.time()
        # sconst = 1
        # # LowTri = np.linalg.cholesky(sconst*currB)
        # # UpTri = LowTri.T
        # for j in range(len(B)):
        #     B_inv[:, j] = sconst * lu_solve(LowTri, UpTri, IDiag[:, j])
        #     #B_inv[:, j] = sconst * scy.linalg.lu_solve((lu, piv),  IDiag[:, j])

        VarB[i] = B_inv * lambMarg[i]
        gamInt[i] = 1 / gams[i] * gamMarg[i]
        meanGamInt[i] = gams[i] * gamMarg[i]
    newCondMean = np.sum(CondResults,0) / theta_scale_O3

    #CondVar = scy.integrate.trapezoid(gamInt) * scy.integrate.trapezoid(VarB.T) / (theta_scale_O3) ** 2
    CondVar = np.sum(gamInt) * np.sum(VarB, 0) / (theta_scale_O3) ** 2

    GamMean =np.sum(meanGamInt)
    #print(GamMean)

    elapsedTime = time.time() - startTime
    print(f'time for whole margFunc incl Mean and CoVar ' + str(index) + f': {elapsedTime}')
    print(f'time for whole margFunc incl Mode Find ' + str(index) + f': {time.time() -TotalStartTime}')
    TotalStartTime
    core = np.zeros((ttTrain.core).size, dtype=np.float64)
    ps_my = 0
    for i in range(0, ttTrain.d):
        cri = ttTrain.core[range(ttTrain.ps[i] - 1, ttTrain.ps[i + 1] - 1)]
        # print(cri)
        #TTCore[i] = cri.reshape((ttTrain.r[i], ttTrain.n[i], ttTrain.r[i + 1]), order='F')
        np.savetxt(index + 'ttTraincoreMargO3' + str(i) + '.txt', cri,
                   header=str(ttTrain.r[i]) + ' ,' + str(ttTrain.n[i]) + ',' + str(ttTrain.r[i + 1]), fmt='%.50f')
        core[range(ps_my, ps_my + ttTrain.r[i] * ttTrain.n[i] * ttTrain.r[i + 1])] = cri
        ps_my = ps_my + ttTrain.r[i] * ttTrain.n[i] * ttTrain.r[i + 1]
        np.savetxt(index +'uniVarGridMargO3'+ str(i) + '.txt', univarGridO3[i])
        np.savetxt(index + 'margPDFMargO3'+ str(i) + '.txt', margPDFO3[i], fmt='%.15f', delimiter='\t')
        if int(ttTrain.r[i + 1]) > maxRank:
            maxRank = int(ttTrain.r[i + 1])



    np.savetxt(index +'GamMean.txt', [GamMean], fmt='%.15f', delimiter='\t')
    np.savetxt(index +'condMean.txt', newCondMean, fmt='%.15f', delimiter='\t')

    np.savetxt(index +'condVar.txt', CondVar, fmt='%.30f', delimiter='\t')

    return GamMean, newCondMean, CondVar, margPDFO3




def log_postTP(params, means, sigmas, A, y, height_values, gamma0):
    n = len(height_values)
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

    #h6Mean = means[6]
    #h6Sigm = sigmas[6]

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

    b0Mean = means[1]
    b0Sigm = sigmas[1]

    h0Mean = means[2]
    h0Sigm = sigmas[2]

    sigmaGrad1 = sigmas[3]
    bmean = means[3]
    #sigmaGrad2 = sigmas[13]
    #sigmaH = sigmas[13]
    sigmaP = sigmas[0]
    pmean = means[0]



    h1 = params[6]
    h2 = params[4]
    h3 = params[10]
    h4 = params[12]
    h5 = params[14]
    #h6 = params[6]
    a0 = params[7]
    a1 = params[5]
    a2 = params[8]
    a3 = params[9]
    a4 = params[11]
    a5 = params[13]
    a6 = params[15]
    b0 = params[1]
    h0 = params[2]
    #b1 = params[12]
    b2 = params[3]
    #h0P =params[13]
    p0 = params[0]
    gam = gamma0#params[15]
    paramT = [h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0]
    paramP = [b2, p0]
    #- ((h6 - h6Mean) / h6Sigm) ** 2
    # postDatT = - gamma0 * np.sum((y - A @ (1 / temp_func(height_values, *paramT).reshape((n, 1)))) ** 2)
    # postDatP = gamma0 * 1e-3 * np.sum((y - A @ pressFunc(height_values[:, 0], *paramP).reshape((n, 1))) ** 2)
    PT = pressFunc(height_values[:, 0], *paramP).reshape((n, 1)) / temp_func(height_values, *paramT).reshape((n, 1))
    #postDat = + SpecNumMeas / 2  * np.log(gam) - 0.5 * gam * np.sum((y - A @ PT ) ** 2)- betaG * gam
    postDat = - 0.5 * gam * np.sum((y - A @ PT) ** 2)

    #postDat = 0
    #- ((popt[0] - b1) / sigmaGrad1) ** 2
    Values =     (- ((h0 - h0Mean) / h0Sigm) ** 2 - ((h1 - h1Mean) / h1Sigm) ** 2 - (
                (h2 - h2Mean) / h2Sigm) ** 2 - (
                        (h3 - h3Mean) / h3Sigm) ** 2 - ((h4 - h4Mean) / h4Sigm) ** 2 - (
                        (h5 - h5Mean) / h5Sigm) ** 2  - ((a0 - a0Mean) / a0Sigm) ** 2 - (
                        (a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2
                - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2- ((a5 - a5Mean) / a5Sigm) ** 2
                  - ((a6 - a6Mean) / a6Sigm) ** 2 - ((b0 - b0Mean) / b0Sigm) ** 2
                 - ((bmean - b2) / sigmaGrad1) ** 2 - (
                            (pmean - p0) / sigmaP) ** 2)
                #- ((means[13] - h0P) / sigmaH) ** 2

    return postDat + 0.5 * Values



def SQ_FullMarg(indices, univarGrid, means, sigmas, L, y, height_values, RealMap, A_lin, AParam, const):
    n = len(height_values)
    m = len(y)

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

    #h6Mean = means[6]
    #h6Sigm = sigmas[6]

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

    #sigmaH = sigmas[13]
    sigmaP = sigmas[3]
    pmean = means[3]
    betaD = 1e-35
    betaG = 1e-35
    gam0 = 2e15
    gamSig = 1e15
    lamb0 = 2000
    lambSig = 1000

    Values = np.zeros(len(indices))
    for j in range(len(indices)):
        gam = univarGrid[0][indices[j, 0].astype(np.int32)]
        lamb = univarGrid[1][indices[j,1].astype(np.int32)]
        h1 = univarGrid[8][indices[j,8].astype(np.int32)]
        h2 = univarGrid[6][indices[j,6].astype(np.int32)]
        h3 = univarGrid[12][indices[j,12].astype(np.int32)]
        h4 = univarGrid[14][indices[j,14].astype(np.int32)]
        h5 = univarGrid[16][indices[j,16].astype(np.int32)]
        a0 = univarGrid[9][indices[j,9].astype(np.int32)]
        a1 = univarGrid[7][indices[j,7].astype(np.int32)]
        a2 = univarGrid[10][indices[j,10].astype(np.int32)]
        a3 = univarGrid[11][indices[j,11].astype(np.int32)]
        a4 = univarGrid[13][indices[j,13].astype(np.int32)]
        a5 = univarGrid[15][indices[j, 15].astype(np.int32)]
        a6 = univarGrid[17][indices[j, 17].astype(np.int32)]
        b0 = univarGrid[4][indices[j,4].astype(np.int32)]
        h0 = univarGrid[3][indices[j, 3].astype(np.int32)]
        b = univarGrid[2][indices[j,2].astype(np.int32)]
        p0 = univarGrid[5][indices[j,5].astype(np.int32)]
        paramT = [h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0]
        paramP = [b, p0]
        P = pressFunc(height_values[:, 0], *paramP).reshape((n, 1))
        T = temp_func(height_values, *paramT).reshape((n, 1))
        #PT = P / T

        CalcA, theatscale = composeAforO3(A_lin, T, P, *AParam)
        CurrA = RealMap @ CalcA
        #CurrA = A * PT.T
        Bp = CurrA.T @ CurrA + lamb * L
        LowTri = np.linalg.cholesky(Bp)
        G = 2 * np.sum(np.log(np.diag(LowTri)))
        #G = g(CurrA, L, lamb)
        ATy = CurrA.T @ y
        B_inv_A_trans_y = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])
        F = f(ATy, y, B_inv_A_trans_y)
        priors = (- ((h0 - h0Mean) / h0Sigm) ** 2 - ((h1 - h1Mean) / h1Sigm) ** 2 - ((h2 - h2Mean) / h2Sigm) ** 2 - (
                (h3 - h3Mean) / h3Sigm) ** 2 - ((h4 - h4Mean) / h4Sigm) ** 2
                  - ((h5 - h5Mean) / h5Sigm) ** 2 - ((a0 - a0Mean) / a0Sigm) ** 2
                  - ((a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2
                  - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2
                  - ((a6 - a6Mean) / a6Sigm) ** 2 - ((a5 - a5Mean) / a5Sigm) ** 2
                  - ((b0 - b0Mean) / b0Sigm) ** 2
                  - ((pmean - p0) / sigmaP) ** 2 - ((bmean - b) / sigmaGrad1) ** 2)  # - 2 * gam[j] * 1e-10
        gamLamPrior = n / 2 * np.log(lamb) + (m / 2 + 1) * np.log(gam) - (betaD * lamb * gam + betaG * gam)
        # gamLamPrior =  n/2 * np.log(lamb) + m/2 * np.log(gam)  - 0.5* ((gam - gam0) / gamSig) ** 2 - 0.5* ((lamb - lamb0) / lambSig) ** 2
        PrevMarg = - 0.5 * G - 0.5 * gam * F

        Values[j] = 0.5 * PrevMarg + 0.25 * priors + 0.5* gamLamPrior
    #CurrVAL =  0.25 * Values + const + priors
    #print(np.exp(np.max(Values + const)))
    #print(np.exp(np.min(Values + const)))
    # print('----')
    return  Values + const


def FullMarg(params, means, sigmas, L, y, height_values, RealMap, A_lin, AParam):
    n = len(height_values)
    m = len(y)

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

    #sigmaH = sigmas[13]
    sigmaP = sigmas[3]
    pmean = means[3]
    betaD = 1e-35
    betaG = 1e-35
    gam0 = 2e15
    gamSig = 1e15
    lamb0 = 2000
    lambSig = 1000


    lamb = params[1]
    gam = params[0]
    h1 = params[8]
    h2 = params[6]
    h3 = params[12]
    h4 = params[14]
    h5 = params[16]
    a0 = params[9]
    a1 = params[7]
    a2 = params[10]
    a3 = params[11]
    a4 = params[13]
    a5 = params[15]
    a6 = params[17]
    b0 = params[4]
    h0 = params[3]
    b = params[2]
    p0 = params[5]
    paramT = [h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0]
    paramP = [b, p0]
    P = pressFunc(height_values[:, 0], *paramP).reshape((n, 1))
    T = temp_func(height_values, *paramT).reshape((n, 1))
    PT = P/T

    CalcA, theatscale = composeAforO3(A_lin, T, P, *AParam)
    CurrA = RealMap @ CalcA
    #CurrA =  CalcA
    #CurrA = A * PT.T
    Bp = CurrA.T @ CurrA + lamb * L
    LowTri = np.linalg.cholesky(Bp)
    G = 2 * np.sum(np.log(np.diag(LowTri)))
    # G = g(CurrA, L, lamb)
    ATy = CurrA.T @ y
    B_inv_A_trans_y = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])
    F = f(ATy, y, B_inv_A_trans_y)
    priors = ( - ((h0 - h0Mean) / h0Sigm) ** 2 -  ((h1 - h1Mean) / h1Sigm) ** 2 - ((h2 - h2Mean) / h2Sigm) ** 2 - (
                   (h3 - h3Mean) / h3Sigm) ** 2 -  ((h4 - h4Mean) / h4Sigm) ** 2
                   - ((h5 - h5Mean) / h5Sigm) ** 2  - ((a0 - a0Mean) / a0Sigm) ** 2
                   - ((a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2
                   - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2
                  - ((a6 - a6Mean) / a6Sigm) ** 2 - ((a5 - a5Mean) / a5Sigm) ** 2
                  - ((b0 - b0Mean) / b0Sigm) ** 2
                   - ((pmean - p0) / sigmaP) ** 2 - ((bmean - b) / sigmaGrad1) ** 2) #- 2 * gam[j] * 1e-10
    gamLamPrior = n/2 * np.log(lamb) + (m/2 + 1) * np.log(gam) -  ( betaD *  lamb * gam + betaG *gam)
    #gamLamPrior =  n/2 * np.log(lamb) + m/2 * np.log(gam)  - 0.5* ((gam - gam0) / gamSig) ** 2 - 0.5* ((lamb - lamb0) / lambSig) ** 2
    PrevMarg =   - 0.5 * G - 0.5 * gam * F

    return  PrevMarg + 0.5 * priors + gamLamPrior


def RTO(A_lin, y, RealMap, L, params, height_values, AParam):
    n = len(height_values)
    lamb = params[1]
    gam = params[0]
    h1 = params[8]
    h2 = params[6]
    h3 = params[12]
    h4 = params[14]
    h5 = params[16]
    a0 = params[9]
    a1 = params[7]
    a2 = params[10]
    a3 = params[11]
    a4 = params[13]
    a5 = params[15]
    a6 = params[17]
    b0 = params[4]
    h0 = params[3]
    b = params[2]
    p0 = params[5]
    paramT = [h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, a5, a6, b0]
    paramP = [b, p0]
    P = pressFunc(height_values[:, 0], *paramP).reshape((n, 1))
    T = temp_func(height_values, *paramT).reshape((n, 1))
    CalcA, theatscale = composeAforO3(A_lin, T, P, *AParam)
    CurrA = RealMap @ CalcA
    ATy = CurrA.T @ y
    ATA = CurrA.T @ CurrA

    W = np.random.normal(loc=0.0, scale=1, size=len(CurrA))
    v_1 = np.sqrt(gam) * CurrA.T @ W

    W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
    v_2 = np.sqrt(gam * lamb) * W2

    SetB = gam * ATA + gam * lamb * L
    RandX = (gam * ATy[:, 0] + v_1 + v_2)

    LowTri = np.linalg.cholesky(SetB)
    UpTri = LowTri.T
    XSampl = lu_solve(LowTri, UpTri, RandX)


    return XSampl/theatscale
    
    
