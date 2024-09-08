from project.utils import cov, vrow
from project.LDA import within_cov
from project.gaussian_density_estimation import get_llr, logpdf_GAU_ND
import numpy as np


def gaussian_classifier(
    DTR, LTR, DVAL, LVAL, prior, type=0, save_cov=False, save_llr=False
):
    """
    type 0: MVG
    type 1: Naive Bayes
    type 2: Tied
    """
    mus = [DTR[:, LTR == i].mean(axis=1) for i in set(LTR)]
    Cs = [cov(DTR[:, LTR == i]) for i in set(LTR)]
    if type == 1:
        Cs = [C * np.identity(C.shape[0]) for C in Cs]
    if type == 2:
        C_w = within_cov(DTR, LTR)
        Cs = [C_w for _ in range(len(mus))]
    densities = np.array(
        [np.exp(logpdf_GAU_ND(DVAL, mus[i], Cs[i])) for i in range(len(mus))]
    )
    SJoint = densities * prior
    SMargin = vrow(SJoint.sum(axis=0))
    SPost = SJoint / SMargin
    if save_cov and save_llr:
        llr = get_llr(DVAL, np.array(mus), np.array(Cs))
        return (
            1 - np.sum(np.argmax(SPost, axis=0) == LVAL) / LVAL.shape[0],
            np.argmax(SPost, axis=0),
            Cs,
            llr
        )
    if save_cov:
        return (
            1 - np.sum(np.argmax(SPost, axis=0) == LVAL) / LVAL.shape[0],
            np.argmax(SPost, axis=0),
            Cs
        )
    if save_llr:
        llr = get_llr(DVAL, np.array(mus), np.array(Cs))
        return (
            1 - np.sum(np.argmax(SPost, axis=0) == LVAL) / LVAL.shape[0],
            np.argmax(SPost, axis=0),
            llr
        )
    return 1 - np.sum(np.argmax(SPost, axis=0) == LVAL) / LVAL.shape[0], np.argmax(
        SPost, axis=0
    )
