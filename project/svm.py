import numpy as np
import scipy
from project.utils import vcol, vrow
from project.bayes_decision_model import optimal_bayes_decision, minimum_detection_cost
import json


def extend_data(D, K):
    return np.vstack([D, np.ones((1, D.shape[1])) * K])


def f(alpha, H):
    loss = 0.5 * np.dot(alpha.T, np.dot(H, alpha)) - np.sum(alpha)
    grad = np.dot(H, alpha).ravel() - np.ones(alpha.size)
    return loss, grad.ravel()


def primalLoss(w, C, D, Z):
    S = np.dot(w.T, D)
    return 0.5 * np.linalg.norm(w) ** 2 + C * np.maximum(0, 1 - Z * S).sum()


def poly_kernelize(D1, D2, c, d):
    return (np.dot(D1.T, D2) + c) ** d


def rbf_kernelize(D1, D2, gamma):
    D1Norms = (D1**2).sum(0)
    D2Norms = (D2**2).sum(0)
    Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
    return np.exp(-gamma * Z)


def svm(
    DTR,
    LTR,
    DVAL,
    LVAL,
    C,
    K,
    kernel=False,
    kernel_type="poly",
    c=0,
    d=1,
    epsilon=1,
    gamma=1,
    pi=0.5,
    debug=False,
    saveLLR=False,
    model_filename_to_save=None,
):
    if kernel:
        DTR_ext = DTR
        if kernel_type == "poly":
            G = poly_kernelize(DTR_ext, DTR_ext, c, d) + epsilon
        else:
            G = rbf_kernelize(DTR_ext, DTR_ext, gamma) + epsilon
        DVAL_ext = DVAL
    else:
        DTR_ext = extend_data(DTR, K)
        G = np.dot(DTR_ext.T, DTR_ext)
        DVAL_ext = extend_data(DVAL, K)
    ZTR = 2 * LTR - 1
    H = G * vcol(ZTR) * vrow(ZTR)
    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        f, np.zeros(DTR.shape[1]), args=(H,), bounds=[(0, C) for i in LTR], factr=1.0
    )
    dualLoss = -f(alphaStar, H)[0]

    w_hat = (alphaStar * ZTR * DTR_ext).sum(1)

    if kernel:
        if kernel_type == "poly":
            GVAL = poly_kernelize(DTR_ext, DVAL_ext, c, d) + epsilon
        else:
            GVAL = rbf_kernelize(DTR_ext, DVAL_ext, gamma) + epsilon
        score = np.dot(alphaStar * ZTR, GVAL)
    else:
        score = np.dot(w_hat.T, DVAL_ext)
    prediction = score > 0
    error_rate = 1 - sum(prediction == LVAL) / LVAL.size
    DCF = optimal_bayes_decision(score, LVAL, pi, 1, 1)[2]
    min_DCF = min(minimum_detection_cost(np.sort(score), score, LVAL, pi, 1, 1)[2])
    primal_loss = -1
    if debug:
        if kernel:
            if kernel_type == "poly":
                print(f"Kernelized SVM with C={C}, c={c}, d={d}, epsilon={epsilon}")
            else:
                print(f"Kernelized SVM with C={C}, c={c}, d={d}, gamma={gamma}")
            print(
                f"Dual loss: {dualLoss}, error rate: {error_rate}, DCF: {DCF}, min_DCF: {min_DCF}"
            )
        else:
            primal_loss = primalLoss(w_hat, C, DTR_ext, ZTR)
            print(f"SVM with C: {C}, K: {K}")
            print(
                f"Primal loss: {primal_loss}, Dual loss: {dualLoss}, Duality gap: {primal_loss - dualLoss}, error rate: {error_rate}, DCF: {DCF}, min_DCF: {min_DCF}"
            )

    if model_filename_to_save is not None:
        model_data = {
            "alphaStar": alphaStar,
            "DTR_ext": DTR_ext,
            "ZTR": ZTR,
            "C": C,
            "gamma": gamma if kernel_type == "rbf" else None,
            "kernel_type": kernel_type,
            "c": c if kernel_type == "poly" else None,
            "d": d if kernel_type == "poly" else None,
        }
        save_svm_model_json(model_data, filename=model_filename_to_save)

    if saveLLR:
        if kernel:
            return -1, dualLoss, error_rate, DCF, min_DCF, score
        else:
            return primal_loss, dualLoss, error_rate, DCF, min_DCF, score
    else:
        if kernel:
            return -1, dualLoss, error_rate, DCF, min_DCF
        else:
            return primal_loss, dualLoss, error_rate, DCF, min_DCF


def save_svm_model_json(model_data, filename="svm_model.json"):
    model_data["alphaStar"] = model_data["alphaStar"].tolist()
    model_data["DTR_ext"] = model_data["DTR_ext"].tolist()
    model_data["ZTR"] = model_data["ZTR"].tolist()

    with open(filename, "w") as f:
        json.dump(model_data, f, indent=4)
    print(f"Model saved {filename}")


def load_svm_model_json(filename):
    with open(filename, "r") as f:
        model_data = json.load(f)

    model_data["alphaStar"] = np.array(model_data["alphaStar"])
    model_data["DTR_ext"] = np.array(model_data["DTR_ext"])
    model_data["ZTR"] = np.array(model_data["ZTR"])

    return model_data


def compute_llr_rbf(model, new_data):
    alphaStar = model["alphaStar"]
    DTR_ext = model["DTR_ext"]
    ZTR = model["ZTR"]
    gamma = model["gamma"]
    kernel_type = model["kernel_type"]

    if kernel_type == "rbf":
        GVAL = rbf_kernelize(DTR_ext, new_data, gamma)
    else:
        raise ValueError("Kernel type must be rbf")

    score = np.dot(alphaStar * ZTR, GVAL)
    return score
