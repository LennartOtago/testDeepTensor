import torch
import deep_tensor as dt

def ln_posterior(x, inv_cov):
    """Compute log_e of posterior of n dimensional multivariate Gaussian.

    Args:

        x: Position at which to evaluate posterior.

    Returns:

        double: Value of posterior at x.

    """
    #-0.5*ndim*np.log(2*np.pi) + 0.5*np.log(np.linalg.det(inv_cov))
    #return  -torch.dot(x,torch.dot(inv_cov,x))/2.0
    return  -torch.matmul(x,torch.matmul(inv_cov,x))/2.0
ndim = 5
# Construct a trivial inverse covariance (identity matrix)
inv_cov = torch.zeros((ndim,ndim))
diag_cov = torch.ones(ndim)

inv_cov.fill_diagonal_(1)


reference = dt.UniformReference()  # define reference measure

bounds = torch.zeros(size=[ndim, 2])
for i in range(0, ndim):
    # currGrid = torch.tensor(np.copy(univarGridFull[i]))
    bounds[i, 0] = -3
    bounds[i, 1] = 3

approximation_domain = bounds  # set grid boundaries
preconditioner = dt.UniformMapping(approximation_domain, reference)  # define preconditioner
gridPoints = 500
basis = dt.Lagrange1(num_elems=gridPoints - 1)  # this is 30 gridpoints piecewise linear interpolation

# here you can choose other interpolation basis such as fourier or chebyshev
bases = dt.ApproxBases(basis, ndim)  # set bases

# rank 1 because variables are uncorrelated
# max_als=2 so that TT-Cross sweeps from left-to-right and right-to-left, max_als = 1 may work as well
# may adjust earlier set options such as ranks number of sweep or allow increase of rank by not specifying tt_method="fixed_rank"
tt_options = dt.TTOptions(max_als=6, init_rank=10,
                          tt_method="fixed_rank")  # set number of sweeps (max_als=1), ranks, fix ranks

tt = dt.TT(tt_options)
ftt = dt.FTT(bases, tt, num_error_samples=0)


def torch_ln_posterior(xs, inv_cov):
    output = torch.zeros(xs.shape[0])
    for i in range(xs.shape[0]):
        output[i] = ln_posterior(xs[i, :], inv_cov)

    return output


neg_log_target = lambda xs: -torch_ln_posterior(xs, inv_cov)


## do TT Aproxximation
# defined above
target_func = dt.TargetFunc(neg_log_target) # set target function
bridge = dt.SingleLayer()  # set single-layer DIRT (i.e., SIRT)
# do DIRT (layered) as  : dirt = dt.DIRT(target_func, preconditioner, ftt)#, bridge) # do single-layer DIRT (i.e., SIRT)
sirt_options = dt.DIRTOptions(num_error_samples=0)
sirt = dt.DIRT(target_func, preconditioner, ftt ,bridge, sirt_options)

from CalcMarg import *
TTCore = [None] * ndim
Grid = [None] * ndim
currEv = [None]
for i in range(0,ndim):
    TTCore[i] =  sirt.sirts[0].ftt.tt.cores[i]
    Grid[i] = torch.linspace(bounds[i,0],bounds[i,1],gridPoints)

absError = 1e-6#e100#/ np.prod()
print('absolute Error: ' + str(absError))
margPDF = getMargfromSQTT(TTCore, Grid, absError)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(ndim, 1)
for i in range(0, ndim):
    axs[i].plot(Grid[i], margPDF[i])
plt.show()
