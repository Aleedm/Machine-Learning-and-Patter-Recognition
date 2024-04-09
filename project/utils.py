import numpy as np
import matplotlib.pyplot as plt


def mcol(v):
    return v.reshape((v.size, 1))


def vcol(v):
    return v.reshape(v.size, 1)


def vrow(v):
    return v.reshape(1, v.size)


def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(",")[0:-1]
                attrs = mcol(np.array([float(i) for i in attrs]))
                DList.append(attrs)
                labelsList.append(line.split(",")[-1].strip())
            except:
                print("Error in line: ", line)
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)


def cov(D):
    mu = D.mean(axis=1)
    DC = D - vcol(mu)
    C = np.dot(DC, DC.T) / D.shape[1]
    return C


def compute_accuracy_error(P, L):
    return (
        np.sum(P == L) / float(L.size),
        np.sum(P != L) / float(L.size),
        np.sum(P == L),
    )


def split(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)
