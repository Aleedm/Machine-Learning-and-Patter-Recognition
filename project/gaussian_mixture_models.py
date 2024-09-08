import numpy as np
import scipy
from project.gaussian_density_estimation import logpdf_GAU_ND
from project.utils import vcol, vrow, diagonal_cov, cov
import json


def logpdf_GMM(x, gmm, saveS=False):
    S = np.array([logpdf_GAU_ND(x, mu, C) + np.log(w) for w, mu, C in gmm])
    if saveS:
        return scipy.special.logsumexp(S, axis=0), S
    return scipy.special.logsumexp(S, axis=0)


def gmm_em(
    x, gmm, threshold=10**-6, cov_type="full", constrain_eigenvalues=False, psi=10**-6
):
    llOld = logpdf_GMM(x, gmm).mean()
    llDelta = None

    while llDelta is None or llDelta > threshold:
        log_density, S = logpdf_GMM(x, gmm, saveS=True)
        log_gamma = np.exp(S - log_density)

        gmmUpd = []
        for gIdx in range(len(gmm)):
            g = log_gamma[gIdx]
            Z = np.sum(g)
            F = vcol((vrow(g) * x).sum(1))
            S = (vrow(g) * x) @ x.T
            m = F / Z
            C = S / Z - m @ m.T

            if cov_type == "diagonal":
                C = diagonal_cov(C)

            w = Z / x.shape[1]
            gmmUpd.append((w, m, C))

        if cov_type == "tied":
            tied_C = 0
            for gIdx in range(len(gmmUpd)):
                w, m, C = gmmUpd[gIdx]
                tied_C += w * C
            gmmUpd = [(w, m, tied_C) for w, m, C in gmmUpd]

        if constrain_eigenvalues:
            for gIdx in range(len(gmmUpd)):
                w, m, C = gmmUpd[gIdx]
                U, s, _ = np.linalg.svd(C)
                s[s < psi] = psi
                C = np.dot(U, np.diag(s) @ U.T)
                gmmUpd[gIdx] = (w, m, C)

        llNew = logpdf_GMM(x, gmmUpd).mean()
        llDelta = llNew - llOld
        llOld = llNew
        gmm = gmmUpd

    return gmm


def get_gmm2(gmm, a=0.1):
    gmmOut = []
    for w, mu, C in gmm:
        U, s, _ = np.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * a
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut


def gmm_lbg_em(
    x,
    components,
    constrain_eigenvalues=False,
    psi=0.1,
    threshold=10**-6,
    cov_type="full",
    a=0.1,
):
    C, mu = cov(x, mu=True)
    mu = vcol(mu)
    if cov_type == "diagonal":
        C = diagonal_cov(C)
    if constrain_eigenvalues:
        U, s, _ = np.linalg.svd(C)
        s[s < psi] = psi
        C = np.dot(U, np.diag(s) @ U.T)
    gmm = [(1.0, mu, C)]

    while len(gmm) < components:
        gmm = get_gmm2(gmm, a)
        gmm = gmm_em(
            x,
            gmm,
            threshold=threshold,
            cov_type=cov_type,
            constrain_eigenvalues=constrain_eigenvalues,
            psi=psi,
        )
    return gmm


import json


def save_gmm_model(gmm0, gmm1, numC0, numC1, cov_type, filepath="gmm_model.json"):
    def gmm_to_dict(gmm):
        return [
            {"weight": w, "mean": mu.tolist(), "covariance": C.tolist()}
            for w, mu, C in gmm
        ]

    model = {
        "gmm0": gmm_to_dict(gmm0),
        "gmm1": gmm_to_dict(gmm1),
        "numC0": numC0,
        "numC1": numC1,
        "cov_type": cov_type,
    }

    with open(filepath, "w") as f:
        json.dump(model, f, indent=4)
    print(f"Model GMM saved {filepath}")


def load_gmm_model(filepath):
    def dict_to_gmm(gmm_dict):
        return [
            (comp["weight"], np.array(comp["mean"]), np.array(comp["covariance"]))
            for comp in gmm_dict
        ]

    with open(filepath, "r") as f:
        model = json.load(f)

    gmm0 = dict_to_gmm(model["gmm0"])
    gmm1 = dict_to_gmm(model["gmm1"])
    return gmm0, gmm1, model["numC0"], model["numC1"], model["cov_type"]


def compute_llr_for_new_data_gmm(gmm0, gmm1, data):
    log_likelihood_class_0 = logpdf_GMM(data, gmm0)
    log_likelihood_class_1 = logpdf_GMM(data, gmm1)
    llr = log_likelihood_class_1 - log_likelihood_class_0
    return llr
