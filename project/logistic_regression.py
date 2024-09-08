import numpy as np
from project.utils import vrow


def f(args):
    y = args[0]
    z = args[1]
    return (y + 3) ** 2 + np.sin(y) + (z + 1) ** 2


def f_compute_gradient(args):
    y = args[0]
    z = args[1]
    value = (y + 3) ** 2 + np.sin(y) + (z + 1) ** 2
    gradient = 2 * (y + 3) + np.cos(y), 2 * (z + 1)
    return value, gradient


def logreg_obj(v, DTR, LTR, l):
    w, b = v[0:-1], v[-1]

    # S = (vcol(w).T @ DTR +b).ravel()
    # S = np.dot(vcol(w).T, DTR) + b
    S = np.dot(w, DTR) + b
    ZTR = 2 * LTR - 1

    loss = np.logaddexp(0, -ZTR * S).mean() + 0.5 * l * np.dot(w, w)

    G = -ZTR / (1 + np.exp(ZTR * S))
    grad_w = np.dot(DTR, G) / DTR.shape[1] + l * w
    grad_b = np.mean(G)
    v = np.hstack([grad_w, grad_b])
    return loss, v


def logreg_obj_wei(v, DTR, LTR, l, pi):
    w, b = v[0:-1], v[-1]

    # S = (vcol(w).T @ DTR +b).ravel()
    # S = np.dot(vcol(w).T, DTR) + b
    S = np.dot(w, DTR) + b
    ZTR = 2 * LTR - 1

    nt = np.sum(LTR == 1)
    nf = np.sum(LTR == 0)

    wT = pi / nt
    wF = (1 - pi) / nf

    loss = np.logaddexp(0, -ZTR * S)
    loss[LTR == 1] *= wT
    loss[LTR == 0] *= wF
    loss = np.sum(loss) + 0.5 * l * np.dot(w, w)

    G = -ZTR / (1 + np.exp(ZTR * S))
    G[LTR == 1] *= wT
    G[LTR == 0] *= wF

    grad_w = (vrow(G) * DTR).sum(1) + l * w.ravel()
    grad_b = np.sum(G)
    v = np.hstack([grad_w, grad_b])
    return loss, v


