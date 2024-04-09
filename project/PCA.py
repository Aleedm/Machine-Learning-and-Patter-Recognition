import numpy as np
from project.utils import cov

def PCA(D, n):
    C = cov(D)
    _, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:n]
    DP = np.dot(P.T, D)
    return DP, P