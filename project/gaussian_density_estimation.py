import numpy as np


def logpdf_GAU_ND_sing(x, mu, C):
    """
    x: data matrix of shape (M, 1)
    mu: numpy array of shape (M, 1)
    C: numpy array of shape (M, M) representing the covariance matrix
    """
    M = x.shape[0]
    C_inv = np.linalg.inv(C)
    sign, slogdet = np.linalg.slogdet(C)
    C_inv, sign, slogdet
    factor1 = -(M / 2) * np.log(2 * np.pi)
    factor2 = -0.5 * slogdet
    factor3 = -0.5 * np.dot(np.dot((x - mu).T, C_inv), (x - mu))

    return factor1 + factor2 + factor3


def logpdf_GAU_ND(x, mu, C):
    """
    x: data matrix of shape (M, n)
    mu: numpy array of shape (M, 1)
    C: numpy array of shape (M, M) representing the covariance matrix
    """
    M = x.shape[0]
    C_inv = np.linalg.inv(C)  # shape (M, M)
    _, slogdet = np.linalg.slogdet(C)
    factor1 = -(M / 2) * np.log(2 * np.pi)
    factor2 = -0.5 * slogdet

    diff = x.T - mu.T  # x.T shape (n, M), mu.T shape (1, M), diff shape (n, M)

    product = np.dot(C_inv, diff.T)
    # C_inv shape (M, M), diff.T shape (M, n), product shape (M, n)
    factor3 = -0.5 * np.sum(diff.T * product, axis=0)
    # diff.T shape (M, n), product shape (M, n), result shape (n,)
    return factor1 + factor2 + factor3


def loglikelihood(x, mu, C):
    return np.sum(logpdf_GAU_ND(x, mu, C))


def get_llr(DVAL, mus, Cs):
    densities = np.array([logpdf_GAU_ND(DVAL, mus[i], Cs[i]) for i in range(len(mus))])
    llr = densities[1] - densities[0]
    return llr