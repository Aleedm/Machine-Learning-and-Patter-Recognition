import numpy as np
import scipy.linalg
from project.utils import vcol, split, compute_accuracy_error
from project.plt import plot_binary_classification_results, plot_hist
from project.PCA import PCA


def within_cov(D, labels):
    classes = set(labels)
    mu = D.mean(axis=1)
    C = np.zeros((D.shape[0], D.shape[0]))
    for class_ in classes:
        Dc = D[:, labels == class_]
        mu_c = Dc.mean(axis=1)
        Dc -= vcol(mu_c)
        C += np.dot(Dc, Dc.T)
    return C / D.shape[1]


def between_cov(D, labels):
    classes = set(labels)
    mu = D.mean(axis=1)
    C = np.zeros((D.shape[0], D.shape[0]))
    for class_ in classes:
        Dc = D[:, labels == class_]
        mu_c = Dc.mean(axis=1)
        C += Dc.shape[1] * np.dot(vcol(mu_c - mu), vcol(mu_c - mu).T)
    return C / D.shape[1]


def LDA(D, labels, m):
    SW = within_cov(D, labels)
    SB = between_cov(D, labels)

    _, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    return np.dot(W.T, D), W


def binary_classification(
    D, L, m, pca=True, debug=True, DTR=None, LTR=None, DVAL=None, LVAL=None, threshold=None, title = None, saveTitle = None
):
    if DTR is None or LTR is None or DVAL is None or LVAL is None:
        (DTR, LTR), (DVAL, LVAL) = split(D, L)
    if pca:
        DTR, P = PCA(DTR, m)
        DVAL = np.dot(P.T, DVAL)

    DTRP_lda, W = LDA(DTR, LTR, 1)
    mean_class_false = DTRP_lda[0, LTR == 0].mean()
    mean_class_true = DTRP_lda[0, LTR == 1].mean()
    if mean_class_false > mean_class_true:
        W = -W
    if threshold is None:
        threshold = (mean_class_false + mean_class_true) / 2
    DVALP_lda = np.dot(W.T, DVAL)
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVALP_lda[0] >= threshold] = 1
    PVAL[DVALP_lda[0] < threshold] = 0
    if debug:
        #plot_binary_classification_results(DVALP_lda[0], threshold, LVAL)
        plot_hist(DVALP_lda[0], LVAL, bins=20, threshold=threshold, title=title, saveTitle=saveTitle)
    return compute_accuracy_error(PVAL, LVAL)
