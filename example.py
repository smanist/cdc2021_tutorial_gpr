"""
This is the pedagogical example for GPR presented in one of the CDC 2021 tutorials.
The example illustrates the adaptive sampling of GPR and effects of kernels on the convergence.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# -----------------------------
# Problem setup
# -----------------------------
XMIN = 0.0
XMAX = 1.0
NSMP = 5      # Number of initial samples
NITR = 30     # Number of iterations

func = lambda x: np.sinc(10*x)      # The function to interpolate

xx = np.linspace(XMIN, XMAX, 501)   # Reference data
yy = func(xx)

# -----------------------------
# Helper functions
# -----------------------------
def fit_model(mdl, x, y):
    """
    Fit a GP using 1D input and output.

    @param mdl: A GaussianProcessRegressor object
    @param x: input, 1D NumPy array
    @param y: output, 1D NumPy array
    """
    mdl.fit(x.reshape(-1,1), y.reshape(-1,1))
    return mdl

def predict(mdl, x):
    """
    Prediction at the new input, with error estimate.
    The error estimate might be a small negative number due to numerical precisions,
    and in this case the estimate is set to zero.

    @param mdl: A trained GaussianProcessRegressor object
    @param x: input, 1D array or scalar
    """
    _y, _s = mdl.predict(np.array([x]).reshape(-1,1), return_std=True)
    _s[_s<0] = 0.0
    return _y.reshape(-1), _s.reshape(-1)

def acquisition_function(mdl, x):
    """
    Acquisition function for adaptive sampling.  Here the error estimate is used.
    Since the optimizer minimizes an objective function, here we return the negative value.

    @param mdl: A trained GaussianProcessRegressor object
    @param x: input, 1D array or scalar
    """
    _, _s = predict(mdl, x)
    return -_s

def adaptive_sampling(mdl):
    """
    A wrapper for the adaptive sampling procedure.  it finds the optimal sample point
    from the acquisition function over the whole parameter space.

    Since the problem is 1D, the optimization is done by grid search with a gradient-based
    refinement.  For higher dimensional problems, one can use, e.g., a gradient-based
    optimizer with multiple random initial guesses, or a "global" optimizer such as the
    differential evolution algorithm, DIRECT (DIviding RECTangle) algorithm, etc.

    @param mdl: A trained GaussianProcessRegressor object
    """
    _s = acquisition_function(mdl, xx)
    _i = np.argmin(_s)
    _r = minimize(lambda x: acquisition_function(mdl, x), xx[_i])
    return _r.x

def error_estimate(mdl, x, y):
    """
    Compute average error over the given truth data.

    @param mdl: A trained GaussianProcessRegressor object
    @param x: input, 1D array
    @param y: truth output, 1D array
    """
    _y, _s = predict(mdl, x)
    _ref = np.max(y) - np.min(y)
    _e = 100*np.linalg.norm(_y-y)/np.sqrt(len(_y))/_ref
    _v = 100*np.linalg.norm(_s)/np.sqrt(len(_s))/_ref
    return _e, _v

ifplt = 1
ifcnv = 0

if ifplt:
    # This section generates Fig. 1
    xs = np.linspace(XMIN, XMAX, NSMP)
    ys = func(xs)
    ker = RBF(0.1, (1e-2, 1e1))

    f, ax = plt.subplots(nrows=4, sharex=True, figsize=(6.4,7.5))
    for _i in range(16):

        # -----------------------------------
        # This is the main block for adaptive sampling
        # -----------------------------------
        # (1) Initialize the GPR object.
        # For a more involved/specialized GPR implementation, there are simpler
        # formulations that augments the GPR with one training sample, without
        # having to update the retrain the hyperparameters.
        # One issue with this current implementation is that, sometimes the
        # training may result in inappropriate hyperparameters and a high prediction
        # error, even after a few adaptive sampling iterations.
        gp = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=9)
        # (2) Train the model using the current samples
        gp = fit_model(gp, xs, ys)
        # (3) Determine where to sample next
        xn = adaptive_sampling(gp)
        # (4) Compute the new sample output; in actual applications, this is
        # the time consuming step.
        yn = func(xn)
        # (5) Append the new sample input and output to the training dataset.
        xs = np.hstack([xs, xn])
        ys = np.hstack([ys, yn])
        # -----------------------------------

        # The following is for plotting.
        if _i%4 == 0:
            yp, sp = predict(gp, xx)
            _e, _ = error_estimate(gp, xx, yy)
            _a = ax[int(_i/4)]
            _a.plot(xx, yy, label='Truth')
            _a.plot(xs[:NSMP], ys[:NSMP], 'ko')
            _a.plot(xs[NSMP:], ys[NSMP:], 'rs', markerfacecolor='none')
            _a.plot(xn, yn, 'rs')
            _a.plot(xx, yp, 'r--', label='GPR prediction')
            _a.fill_between(xx, yp-2*sp, yp+2*sp, color='g', alpha=0.3, label='Uncertainty of $2\sigma$')
            _a.set_ylabel('y', fontsize=14)
            _a.set_title(f'Iter {_i+1}: Error {_e:3.2f}%')
            _a.tick_params(labelsize=12)
    ax[-1].set_xlabel('x', fontsize=14)
    ax[-1].legend()

    f.savefig('as_iters.png', dpi=600, bbox_inches='tight', transparent=True)

if ifcnv:
    # This section generates Fig. 2
    # The core part of the code is the same as the previous one.
    # We just apply it with different kernels, record the errors at each iteration,
    # and plot the convergence history.
    kers = [
        RBF(1.0, (1e-2, 1e2)),
        Matern(1.0, (1e-2, 1e2), nu=2.5),
        Matern(1.0, (1e-2, 1e2), nu=1.5)
    ]

    errs = np.zeros((4,2,NITR))
    for _j, _k in enumerate(kers):
        xs = np.linspace(XMIN, XMAX, NSMP)
        ys = func(xs)

        for _i in range(NITR):
            gp = GaussianProcessRegressor(kernel=_k, n_restarts_optimizer=9)
            gp = fit_model(gp, xs, ys)
            yp, sp = predict(gp, xx)

            xn = adaptive_sampling(gp)
            yn = func(xn)
            xs = np.hstack([xs, xn])
            ys = np.hstack([ys, yn])

            _e, _v = error_estimate(gp, xx, yy)
            errs[_j, :, _i] = [_e, _v]

    clrs = ['b', 'r', 'g']
    lbls = ['SE', 'Matern 5/2', 'Matern 3/2']
    idx = np.arange(NITR)+1
    f = plt.figure()
    for _j, _k in enumerate(kers):
        _e = errs[_j, 0]
        _v = errs[_j, 1]
        plt.semilogy(idx, _e, clrs[_j]+'-', label=lbls[_j])
        # plt.fill_between(idx, _e-2*_v, _e+2*_v, color=clrs[_j], alpha=0.3)
    plt.ylim([-5,20])
    plt.xlabel('No. of new samples', fontsize=14)
    plt.ylabel('NRMSE, %', fontsize=14)
    plt.legend()
    ax = plt.gca()
    ax.tick_params(labelsize=12)

    f.savefig('as_conv.png', dpi=600, bbox_inches='tight', transparent=True)

plt.show()