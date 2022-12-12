import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass


@dataclass
class SingleBranchNoHiddenLayerParams:
    w0: np.array
    b0: int
    w1: int


@dataclass
class Data:
    X: np.array
    y: np.array


def lrelu(x):
    return np.where(x > 0, x, x * 0.01)


def dlreludx(x):
    return np.where(x > 0, 1, 0.01)


def lrelu_predict(p: SingleBranchNoHiddenLayerParams, d: Data):
    return lrelu(d.X @ p.w0 + p.b0) * p.w1


def lrelu_drssdw0(p: SingleBranchNoHiddenLayerParams, d: Data):
    z0 = d.X @ p.w0 + p.b0
    a0 = lrelu(z0)
    y_hat = a0 * p.w1
    return (2 * (y_hat - d.y) * p.w1 * dlreludx(z0)) @ d.X


def lrelu_rss(p: SingleBranchNoHiddenLayerParams, d: Data):
    return np.sum((lrelu_predict(p, d) - d.y) ** 2)


def tanh_predict(p: SingleBranchNoHiddenLayerParams, d: Data):
    return np.tanh(d.X @ p.w0 + p.b0) * p.w1


def tanh_drssdw0(p: SingleBranchNoHiddenLayerParams, d: Data):
    z0 = d.X @ p.w0 + p.b0
    a0 = np.tanh(z0)
    y_hat = a0 * p.w1
    return (2 * (y_hat - d.y) * p.w1 * (1 - np.tanh(z0) ** 2)) @ d.X


def tanh_rss(p: SingleBranchNoHiddenLayerParams, d: Data):
    return np.sum((tanh_predict(p, d) - d.y) ** 2)


def tanh_posterior(p: SingleBranchNoHiddenLayerParams, d: Data, lambda_w0: float, lambda_e: float, w0_ix: int):
    return -lambda_e / 2 * tanh_rss(p, d) - lambda_w0 / 2 * p.w0[w0_ix] ** 2


def plot_posterior(p: SingleBranchNoHiddenLayerParams, d1: Data, d2: Data, w0_ix: int, posterior, margin=100):
    w0_val = p.w0[w0_ix]

    w0s = np.linspace(w0_val - margin, w0_val + margin, 100)
    logl_tst = []
    logl_trn = []

    for v in w0s:
        p.w0[w0_ix] = v
        logl_tst.append(posterior(p, d1, 1, 1, w0_ix))
        logl_trn.append(posterior(p, d2, 1, 1, w0_ix))

    p.w0[w0_ix] = w0_val

    plt.plot(w0s, logl_tst, label='D1')
    plt.plot(w0s, logl_trn, label='D2')
    plt.legend()
    plt.axvline(w0_val, linestyle=':')


def sim_single_branch_no_hidden_layer(n: int, m: int, h2: float, plot=False):
    # tanh default
    predict = tanh_predict
    drssdw0 = tanh_drssdw0

    rng = np.random.default_rng()
    mafs = rng.uniform(0.01, 0.5, m)
    X = np.array([rng.binomial(2, maf, n) for maf in mafs]).T

    w0 = rng.normal(0, 0.5, m)
    b0 = rng.normal(0, 0.5)
    w1 = 5

    p = SingleBranchNoHiddenLayerParams(w0, b0, w1)

    g = predict(p, Data(X, None))
    var_g = g.var()

    var_p = var_g / h2
    var_e = var_p - var_g
    y = g + rng.normal(0, np.sqrt(var_e), n)

    n1 = int(n / 2)

    X1 = X[:n1, :]
    X2 = X[n1:, :]
    y1 = y[:n1]
    y2 = y[n1:]

    d1 = Data(X1, y1)
    d2 = Data(X2, y2)

    if plot:
        plt.figure()
        plt.plot(y, g, '.')

        plt.figure()
        plt.plot(g, '.')

        plt.figure()
        plt.plot(y1, '.')
        plt.plot(y2, '.')

        plt.figure()
        plt.plot(y1, g[:n1], '.')

        plt.plot(y2, g[n1:], '.')

        plt.figure()
        plt.title('means')
        plt.plot(X2.mean(axis=0), X1.mean(axis=0), '.')
        plt.ylabel('x1')
        plt.xlabel('x2')

        plt.figure()
        plt.title('vars')
        plt.plot(X2.var(axis=0), X1.var(axis=0), '.')
        plt.ylabel('x1')
        plt.xlabel('x2')

        plt.figure()
        plt.title('drssdw0')
        plt.plot(drssdw0(p, d2), drssdw0(p, d1), ".")
        plt.ylabel('x1')
        plt.xlabel('x2')

    return d1, d2, p
