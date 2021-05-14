import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

XMIN = 0.0
XMAX = 1.0
NSMP = 5
NITR = 30

func = lambda x: np.sinc(10*x)
xx = np.linspace(XMIN, XMAX, 501)
yy = func(xx)

def fit_model(mdl, x, y):
    mdl.fit(x.reshape(-1,1), y.reshape(-1,1))
    return mdl

def predict(mdl, x):
    _y, _s = mdl.predict(np.array([x]).reshape(-1,1), return_std=True)
    _s[_s<0] = 0.0
    return _y.reshape(-1), _s.reshape(-1)

def acquisition_function(mdl, x):
    _, _s = predict(mdl, x)
    return -_s

def adaptive_sampling(mdl):
    _s = acquisition_function(mdl, xx)
    _i = np.argmin(_s)
    _r = minimize(lambda x: acquisition_function(mdl, x), xx[_i])
    return _r.x

def error_estimate(mdl, x, y):
    _y, _s = predict(mdl, x)
    _ref = np.max(y) - np.min(y)
    _e = 100*np.linalg.norm(_y-y)/np.sqrt(len(_y))/_ref
    _v = 100*np.linalg.norm(_s)/np.sqrt(len(_s))/_ref
    return _e, _v

ifplt = 1
ifcnv = 0

if ifplt:
    xs = np.linspace(XMIN, XMAX, NSMP)
    ys = func(xs)
    ker = RBF(0.1, (1e-2, 1e1))

    f, ax = plt.subplots(nrows=4, sharex=True, figsize=(6.4,7.5))
    for _i in range(16):
        gp = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=9)
        gp = fit_model(gp, xs, ys)
        yp, sp = predict(gp, xx)

        xn = adaptive_sampling(gp)
        yn = func(xn)
        xs = np.hstack([xs, xn])
        ys = np.hstack([ys, yn])

        if _i%4 == 0:
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