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


def cov(D, mu=False):
    mu_ = D.mean(axis=1)
    DC = D - vcol(mu_)
    C = np.dot(DC, DC.T) / D.shape[1]
    if mu:
        return C, mu_
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


def confusion_matrix(true_labels, predicted_labels):
    classes = np.unique(np.concatenate((true_labels, predicted_labels)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    label_to_index = {label: idx for idx, label in enumerate(classes)}

    for true, pred in zip(true_labels, predicted_labels):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        matrix[pred_index, true_index] += 1

    return matrix


def diagonal_cov(C):
    return C * np.eye(C.shape[0])


def get_pearson_correlation_coefficient(C):
    return C / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5))


def quadratic_feature_expansion(D):
    D_T = D.T
    D_expanded = []
    for d in D_T:
        outer_product = np.outer(d, d).flatten()
        expanded_feature = np.concatenate([outer_product, d])
        D_expanded.append(expanded_feature)
    D_expanded = np.array(D_expanded).T

    return D_expanded


def extend_features(D):
    quadratic_features = D**2
    return np.vstack([quadratic_features, D])


def extract_train_val_folds_from_ary(X, idx, KFOLD=5):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]