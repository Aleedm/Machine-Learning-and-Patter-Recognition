from project.utils import confusion_matrix
import numpy as np


def DCF(conf_matrix, pi, Cfn, Cfp):
    Pfn = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1])
    Pfp = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[0, 0])
    return pi * Cfn * Pfn + (1 - pi) * Cfp * Pfp


def norm_DCF(conf_matrix, pi, Cfn, Cfp):
    return DCF(conf_matrix, pi, Cfn, Cfp) / min(pi * Cfn, (1 - pi) * Cfp)


def optimal_bayes_decision(llr, labels, pi, Cfn, Cfp):
    threshold = -np.log((pi * Cfn) / ((1 - pi) * Cfp))
    decision = llr > threshold
    c_f = confusion_matrix(labels, decision)
    dcf = DCF(c_f, pi, Cfn, Cfp)
    norm_dcf = norm_DCF(c_f, pi, Cfn, Cfp)
    return c_f, dcf, norm_dcf


def minimum_detection_cost(threshold, llr, labels, pi, Cfn, Cfp):
    dcfs = []
    norm_dcfs = []
    c_fs = []
    for t in threshold:
        decision = llr > t
        c_f = confusion_matrix(labels, decision)
        dcf = DCF(c_f, pi, Cfn, Cfp)
        norm_dcf = norm_DCF(c_f, pi, Cfn, Cfp)
        c_fs.append(c_f)
        dcfs.append(dcf)
        norm_dcfs.append(norm_dcf)
    return c_fs, dcfs, norm_dcfs


def get_effective_prior(prior, Cfn, Cfp):
    return (Cfn * prior) / (Cfn * prior + Cfp * (1 - prior))


def dcf_mindcf(llr, LVAL, threshold, pi, cfn, cfp, title="", debug=False):
    c_f, _, dcf = optimal_bayes_decision(llr, LVAL, pi, cfn, cfp)
    _, _, dcfs = minimum_detection_cost(threshold, llr, LVAL, pi, cfn, cfp)
    min_dcf = min(dcfs)
    print(f"{title}, DCF: {dcf} minDCF: {min_dcf}")
    if debug:
        print(f"Cost: {c_f}")
