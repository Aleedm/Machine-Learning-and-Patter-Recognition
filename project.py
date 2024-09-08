import numpy as np
import pandas as pd
from project.gaussian_mixture_models import (
    compute_llr_for_new_data_gmm,
    gmm_lbg_em,
    load_gmm_model,
    logpdf_GMM,
    save_gmm_model,
)
from project.svm import compute_llr_rbf, load_svm_model_json, svm
from project.bayes_decision_model import (
    dcf_mindcf,
    get_effective_prior,
    minimum_detection_cost,
    optimal_bayes_decision,
)
from project.gaussian_classifier import gaussian_classifier
from project.utils import (
    extract_train_val_folds_from_ary,
    load,
    quadratic_feature_expansion,
    split,
    load,
    get_pearson_correlation_coefficient,
    vcol,
    vrow,
)
from project.plt import (
    bayes_error_plot,
    bayes_error_plot_multiple,
    plot_DCF_minDCF,
    plot_hist,
    plot_scatter,
    plot_gau_den_est,
)
from project.PCA import PCA
from project.LDA import LDA, binary_classification
import scipy
from project.logistic_regression import logreg_obj, logreg_obj_wei
import matplotlib.pyplot as plt


def lab_2():
    print("\nLab 2 - Feature analysis")
    data, labels = load("./project/train.csv")
    for i in range(data.shape[0]):
        plot_hist(
            data[i],
            labels,
            bins=20,
            saveTitle=f"./lab_2_feature_hist{i+1}.png",
            title=f"Feature {i+1}",
        )
        print(
            f"Mean of feature {i+1} \nfor class 0: {data[i][labels == 0].mean()} \nclass 1: {data[i][labels == 1].mean()}"
        )
        print(
            f"Variance of feature {i+1} \nfor class 0: {data[i][labels == 0].var()} \nclass 1: {data[i][labels == 1].var()}"
        )

    feature_pairs = [(0, 1), (2, 3), (4, 5)]
    for i, j in feature_pairs:
        plot_scatter(
            data[i],
            f"Feature {i+1}",
            data[j],
            f"Feature {j+1}",
            labels,
            saveTitle=f"./lab_2_feature_scatter{i+1}_{j+1}.png",
        )


def lab_3():
    print("\nLab 3 - Dimensionality reduction")
    data, labels = load("./project/train.csv")
    pca = PCA(data, 6)
    for i in range(6):
        plot_hist(
            pca[0][i],
            labels,
            20,
            title=f"PCA {i+1}",
            saveTitle=f"./lab_3_pca_hist{i+1}.png",
        )
    lda, _ = LDA(data, labels, 1)
    plot_hist(lda[0], labels, 20, title=f"LDA", saveTitle=f"./lab_3_lda_hist.png")
    accuracy, error, _ = binary_classification(
        data,
        labels,
        1,
        pca=False,
        saveTitle="./lab_3_binary_classification.png",
        title="Binary classification",
    )
    print(f"LDA accuracy: {accuracy}, error: {error}")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        accuracy, error, _ = binary_classification(
            data,
            labels,
            1,
            pca=False,
            threshold=threshold,
            saveTitle=f"./lab_3_binary_classification_threshold_{threshold}.png",
            title=f"Binary classification with threshold {threshold}",
        )
        print(f"LDA accuracy with threshold {threshold}: {accuracy}, error: {error}")
    for i in range(6):
        accuracy, error, _ = binary_classification(
            data,
            labels,
            i + 1,
            pca=True,
            saveTitle=f"./lab_3_binary_classification_pca_{i+1}.png",
            title=f"Binary classification with PCA {i+1}",
        )
        print(f"PCA accuracy for feature {i+1}: {accuracy}, error: {error}")


def lab_4():
    print("\nLab 4 - Gaussian density estimation")
    data, labels = load("./project/train.csv")
    for i in range(6):
        D_t = data[:, labels == 1][i]
        D_f = data[:, labels == 0][i]
        plot_gau_den_est(
            D_t,
            D_f,
            saveTitle=f"./lab_4_gau_den_est_feature_{i+1}.png",
            title=f"Feature {i+1}",
        )


def lab_5():
    print("\nLab 5 - Bayesian decision theory")
    data, labels = load("./project/train.csv")
    (DTR, LTR), (DVAL, LVAL) = split(data, labels)
    err_mvg, _, Cs_mvg = gaussian_classifier(
        DTR, LTR, DVAL, LVAL, 1 / 2, 0, save_cov=True
    )
    _, err_lda, _ = binary_classification(
        data, labels, 1, pca=False, debug=False, DTR=DTR, LTR=LTR, DVAL=DVAL, LVAL=LVAL
    )
    print(f"mvg - error: {err_mvg}\nlda - error: {err_lda}")
    err_tied, _, _ = gaussian_classifier(DTR, LTR, DVAL, LVAL, 1 / 2, 2, save_cov=True)
    print(f"tied - error: {err_tied}")
    err_naive_bayes, _, _ = gaussian_classifier(
        DTR, LTR, DVAL, LVAL, 1 / 2, 1, save_cov=True
    )
    print(f"naive bayes - error: {err_naive_bayes}")

    print("\nCovariance matrix for mvg - class 0:")
    print(pd.DataFrame(Cs_mvg[0]))
    print("\nCovariance matrix for mvg - class 1:")
    print(pd.DataFrame(Cs_mvg[1]))

    print("\n Pearson correlation coefficient for mvg - class 0:")
    print(pd.DataFrame(get_pearson_correlation_coefficient(Cs_mvg[0])))
    print("\n Pearson correlation coefficient for mvg - class 1:")
    print(pd.DataFrame(get_pearson_correlation_coefficient(Cs_mvg[1])))

    DTR_1_4 = DTR[0:4]
    DVAL_1_4 = DVAL[0:4]
    err_mvg_1_4, _, _ = gaussian_classifier(
        DTR_1_4, LTR, DVAL_1_4, LVAL, 1 / 2, 0, save_cov=True
    )
    err_naive_bayes_1_4, _, _ = gaussian_classifier(
        DTR_1_4, LTR, DVAL_1_4, LVAL, 1 / 2, 1, save_cov=True
    )
    err_tied_1_4, _, _ = gaussian_classifier(
        DTR_1_4, LTR, DVAL_1_4, LVAL, 1 / 2, 2, save_cov=True
    )
    _, err_lda_1_4, _ = binary_classification(
        data,
        labels,
        1,
        pca=False,
        debug=False,
        DTR=DTR_1_4,
        LTR=LTR,
        DVAL=DVAL_1_4,
        LVAL=LVAL,
    )
    print(
        f"\nMVG only 1-4 features: {err_mvg_1_4}\nnaive bayes only 1-4 features: {err_naive_bayes_1_4}\ntied only 1-4 features: {err_tied_1_4}\nlda only 1-4 features: {err_lda_1_4}"
    )

    DTR_1_2 = DTR[0:2]
    DVAL_1_2 = DVAL[0:2]
    err_mvg_1_2, _, _ = gaussian_classifier(
        DTR_1_2, LTR, DVAL_1_2, LVAL, 1 / 2, 0, save_cov=True
    )
    err_naive_bayes_1_2, _, _ = gaussian_classifier(
        DTR_1_2, LTR, DVAL_1_2, LVAL, 1 / 2, 1, save_cov=True
    )
    err_tied_1_2, _, _ = gaussian_classifier(
        DTR_1_2, LTR, DVAL_1_2, LVAL, 1 / 2, 2, save_cov=True
    )
    _, err_lda_1_2, _ = binary_classification(
        data,
        labels,
        1,
        pca=False,
        debug=False,
        DTR=DTR_1_2,
        LTR=LTR,
        DVAL=DVAL_1_2,
        LVAL=LVAL,
    )
    print(
        f"\nMVG only 1-2 features: {err_mvg_1_2}\nnaive bayes only 1-2 features: {err_naive_bayes_1_2}\ntied only 1-2 features: {err_tied_1_2}\nlda only 1-2 features: {err_lda_1_2}"
    )

    DTR_3_4 = DTR[2:4]
    DVAL_3_4 = DVAL[2:4]
    err_mvg_3_4, _, _ = gaussian_classifier(
        DTR_3_4, LTR, DVAL_3_4, LVAL, 1 / 2, 0, save_cov=True
    )
    err_naive_bayes_3_4, _, _ = gaussian_classifier(
        DTR_3_4, LTR, DVAL_3_4, LVAL, 1 / 2, 1, save_cov=True
    )
    err_tied_3_4, _, _ = gaussian_classifier(
        DTR_3_4, LTR, DVAL_3_4, LVAL, 1 / 2, 2, save_cov=True
    )
    _, err_lda_3_4, _ = binary_classification(
        data,
        labels,
        1,
        pca=False,
        debug=False,
        DTR=DTR_3_4,
        LTR=LTR,
        DVAL=DVAL_3_4,
        LVAL=LVAL,
    )
    print(
        f"\nMVG only 3-4 features: {err_mvg_3_4}\nnaive bayes only 3-4 features: {err_naive_bayes_3_4}\ntied only 3-4 features: {err_tied_3_4}\nlda  only 3-4 features: {err_lda_3_4}"
    )

    for i in range(1, 7):
        data_pca = PCA(data, i)[0]
        (DTR_pca, LTR), (DVAL_pca, LVAL) = split(data_pca, labels)

        err_mvg, _, Cs_mvg = gaussian_classifier(
            DTR_pca, LTR, DVAL_pca, LVAL, 1 / 2, 0, save_cov=True
        )
        err_naive_bayes, _, _ = gaussian_classifier(
            DTR_pca, LTR, DVAL_pca, LVAL, 1 / 2, 1, save_cov=True
        )
        err_tied, _, _ = gaussian_classifier(
            DTR_pca, LTR, DVAL_pca, LVAL, 1 / 2, 2, save_cov=True
        )

        _, err_lda, _ = binary_classification(
            data,
            labels,
            1,
            pca=False,
            debug=False,
            DTR=DTR_pca,
            LTR=LTR,
            DVAL=DVAL_pca,
            LVAL=LVAL,
        )

        print(
            f"\nMVG PCA{i}: {err_mvg}\nnaive bayes PCA{i}: {err_naive_bayes}\ntied PCA{i}: {err_tied}\nlda-PCA: {err_lda}"
        )


def lab_7():
    print("\nLab 7 - Minimum detection cost")
    params = [
        (0.5, 1.0, 1.0, "Application 1: prior = 0.5, Cfn = 1.0, Cfp = 1.0"),
        (0.9, 1.0, 1.0, "Application 2: prior = 0.9, Cfn = 1.0, Cfp = 1.0"),
        (0.1, 1.0, 1.0, "Application 3: prior = 0.1, Cfn = 1.0, Cfp = 1.0"),
        (0.5, 1.0, 9.0, "Application 4: prior = 0.5, Cfn = 1.0, Cfp = 9.0"),
        (0.5, 9.0, 1.0, "Application 5: prior = 0.5, Cfn = 9.0, Cfp = 1.0"),
    ]

    for prior, Cfn, Cfp, description in params:
        effective_prior = get_effective_prior(prior, Cfn, Cfp)
        print(f"{description}: effective prior = {effective_prior}, prior = {prior}")

    data, labels = load("./project/train.csv")
    (DTR, LTR), (DVAL, LVAL) = split(data, labels)

    _, _, llr_mvg = gaussian_classifier(DTR, LTR, DVAL, LVAL, 1 / 2, 0, save_llr=True)
    threshold_mvg = np.sort(llr_mvg)
    _, _, llr_bayes = gaussian_classifier(DTR, LTR, DVAL, LVAL, 1 / 2, 1, save_llr=True)
    threshold_bayes = np.sort(llr_bayes)
    _, _, llr_tied = gaussian_classifier(DTR, LTR, DVAL, LVAL, 1 / 2, 2, save_llr=True)
    threshold_tied = np.sort(llr_tied)

    DTR_pca, P = PCA(DTR, 4)
    DVAL_pca = np.dot(P.T, DVAL)
    _, _, llr_mvg_pca = gaussian_classifier(
        DTR_pca, LTR, DVAL_pca, LVAL, 1 / 2, 0, save_llr=True
    )
    threshold_mvg_pca = np.sort(llr_mvg_pca)
    _, _, llr_bayes_pca = gaussian_classifier(
        DTR_pca, LTR, DVAL_pca, LVAL, 1 / 2, 1, save_llr=True
    )
    threshold_bayes_pca = np.sort(llr_bayes_pca)
    _, _, llr_tied_pca = gaussian_classifier(
        DTR_pca, LTR, DVAL_pca, LVAL, 1 / 2, 2, save_llr=True
    )
    threshold_tied_pca = np.sort(llr_tied_pca)
    effPriorLogOdds = np.linspace(-3, 3, 21)
    pis = 1 / (1 + np.exp(-effPriorLogOdds))

    pi = 0.5
    cfn = 1
    cfp = 1
    print(f"Problem: pi = {pi}, cfn = {cfn}, cfp = {cfp}")
    dcf_mindcf(llr_mvg, LVAL, threshold_mvg, pi, cfn, cfp, "MVG")
    dcf_mindcf(llr_bayes, LVAL, threshold_bayes, pi, cfn, cfp, "Bayes")
    dcf_mindcf(llr_tied, LVAL, threshold_tied, pi, cfn, cfp, "tied")
    dcf_mindcf(llr_mvg_pca, LVAL, threshold_mvg_pca, pi, cfn, cfp, "MVG_pca_4")
    dcf_mindcf(llr_bayes_pca, LVAL, threshold_bayes_pca, pi, cfn, cfp, "Bayes_pca_4")
    dcf_mindcf(llr_tied_pca, LVAL, threshold_tied_pca, pi, cfn, cfp, "tied_pca_4")

    pi = 0.9
    cfn = 1
    cfp = 1
    print(f"\nProblem: pi = {pi}, cfn = {cfn}, cfp = {cfp}")
    dcf_mindcf(llr_mvg, LVAL, threshold_mvg, pi, cfn, cfp, "MVG")
    dcf_mindcf(llr_bayes, LVAL, threshold_bayes, pi, cfn, cfp, "Bayes")
    dcf_mindcf(llr_tied, LVAL, threshold_tied, pi, cfn, cfp, "tied")
    dcf_mindcf(llr_mvg_pca, LVAL, threshold_mvg_pca, pi, cfn, cfp, "MVG_pca_4")
    dcf_mindcf(llr_bayes_pca, LVAL, threshold_bayes_pca, pi, cfn, cfp, "Bayes_pca_4")
    dcf_mindcf(llr_tied_pca, LVAL, threshold_tied_pca, pi, cfn, cfp, "tied_pca_4")

    pi = 0.1
    cfn = 1
    cfp = 1
    print(f"\nProblem: pi = {pi}, cfn = {cfn}, cfp = {cfp}")
    dcf_mindcf(llr_mvg, LVAL, threshold_mvg, pi, cfn, cfp, "MVG")
    dcf_mindcf(llr_bayes, LVAL, threshold_bayes, pi, cfn, cfp, "Bayes")
    dcf_mindcf(llr_tied, LVAL, threshold_tied, pi, cfn, cfp, "tied")
    dcf_mindcf(llr_mvg_pca, LVAL, threshold_mvg_pca, pi, cfn, cfp, "MVG_pca_4")
    dcf_mindcf(llr_bayes_pca, LVAL, threshold_bayes_pca, pi, cfn, cfp, "Bayes_pca_4")
    dcf_mindcf(llr_tied_pca, LVAL, threshold_tied_pca, pi, cfn, cfp, "tied_pca_4")

    effPriorLogOdds = np.linspace(-4, 4, 21)
    pis = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    bayes_error_plot(
        effPriorLogOdds,
        llr_mvg_pca,
        LVAL,
        threshold_mvg_pca,
        pis,
        1.0,
        1.0,
        title="MVG - PCA 4",
        saveTitle="./lab_7_mvg_pca_4.png",
    )
    bayes_error_plot(
        effPriorLogOdds,
        llr_bayes_pca,
        LVAL,
        threshold_bayes_pca,
        pis,
        1.0,
        1.0,
        title="Bayes - PCA 4",
        saveTitle="./lab_7_bayes_pca_4.png",
    )
    bayes_error_plot(
        effPriorLogOdds,
        llr_tied_pca,
        LVAL,
        threshold_tied_pca,
        pis,
        1.0,
        1.0,
        title="Tied - PCA 4",
        saveTitle="./lab_7_tied_pca_4.png",
    )


def lab_8():
    print("\nLab 8 - Logistic regression")
    print("Full data")
    data, labels = load("./project/train.csv")
    (DTR, LTR), (DVAL, LVAL) = split(data, labels)
    log_spaced_lambda = np.logspace(-4, 2, 13)
    x0 = np.zeros(DTR.shape[0] + 1)
    pi = 0.1
    vfs, j = zip(
        *[
            (result[0], result[1])
            for result in [
                scipy.optimize.fmin_l_bfgs_b(
                    logreg_obj, x0=x0, args=(DTR, LTR, l), approx_grad=False
                )
                for l in log_spaced_lambda
            ]
        ]
    )
    ws = [vf[:-1] for vf in vfs]
    bs = [vf[-1] for vf in vfs]
    SVals = [np.dot(w.T, DVAL) + b for w, b in zip(ws, bs)]
    LPs = [SVal > 0 for SVal in SVals]
    for i, LP in enumerate(LPs):
        error_rate = round(1 - (sum(LP == LVAL) / len(LVAL)), 4)
        lambda_val = round(log_spaced_lambda[i], 4)
        print(f"Error rate: {error_rate}, lambda: {lambda_val}")
    emp_p = sum(LTR == 1) / len(LTR)
    print(f"Empirical prior: {emp_p}")
    llrs = SVals - np.log(emp_p / (1 - emp_p))
    DCFs = [optimal_bayes_decision(llr, LVAL, pi, 1, 1)[2] for llr in llrs]
    minDCFs = [
        min(minimum_detection_cost(np.sort(llr), llr, LVAL, pi, 1, 1)[2])
        for llr in llrs
    ]
    min_minDCF = min(minDCFs)
    min_lambda = log_spaced_lambda[minDCFs.index(min_minDCF)]
    print(f"Best minDCF: {min_minDCF} for lambda: {min_lambda}")
    plot_DCF_minDCF(
        log_spaced_lambda,
        DCFs,
        minDCFs,
        saveTitle="./lab_8_full.png",
        title="Full data - DCF and MinDCF vs Lambda",
    )

    print("\nReduced data")
    log_spaced_lambda = np.logspace(-4, 2, 13)
    (DTR, LTR), (DVAL, LVAL) = split(data, labels)
    DTR = DTR[:, ::50]
    LTR = LTR[::50]
    x0 = np.zeros(DTR.shape[0] + 1)
    pi = 0.1
    vfs, j = zip(
        *[
            (result[0], result[1])
            for result in [
                scipy.optimize.fmin_l_bfgs_b(
                    logreg_obj, x0=x0, args=(DTR, LTR, l), approx_grad=False
                )
                for l in log_spaced_lambda
            ]
        ]
    )
    ws = [vf[:-1] for vf in vfs]
    bs = [vf[-1] for vf in vfs]
    SVals = [np.dot(w.T, DVAL) + b for w, b in zip(ws, bs)]
    LPs = [SVal > 0 for SVal in SVals]
    for i, LP in enumerate(LPs):
        error_rate = round(1 - (sum(LP == LVAL) / len(LVAL)), 4)
        lambda_val = round(log_spaced_lambda[i], 4)
        print(f"Error rate: {error_rate}, lambda: {lambda_val}")
    emp_p = sum(LTR == 1) / len(LTR)
    print(f"Empirical prior: {emp_p}")
    llrs = SVals - np.log(emp_p / (1 - emp_p))
    DCFs = [optimal_bayes_decision(llr, LVAL, pi, 1, 1)[2] for llr in llrs]
    minDCFs = [
        min(minimum_detection_cost(np.sort(llr), llr, LVAL, pi, 1, 1)[2])
        for llr in llrs
    ]
    min_minDCF = min(minDCFs)
    min_lambda = log_spaced_lambda[minDCFs.index(min_minDCF)]
    print(f"Best minDCF: {min_minDCF} for lambda: {min_lambda}")
    plot_DCF_minDCF(
        log_spaced_lambda,
        DCFs,
        minDCFs,
        saveTitle="./lab_8_reduced.png",
        title="Reduced data - DCF and MinDCF vs Lambda",
    )

    print("\nWeighted data")
    log_spaced_lambda = np.logspace(-4, 2, 13)
    (DTR, LTR), (DVAL, LVAL) = split(data, labels)
    x0 = np.zeros(DTR.shape[0] + 1)
    pi = 0.1
    vfs, j = zip(
        *[
            (result[0], result[1])
            for result in [
                scipy.optimize.fmin_l_bfgs_b(
                    logreg_obj_wei, x0=x0, args=(DTR, LTR, l, pi), approx_grad=False
                )
                for l in log_spaced_lambda
            ]
        ]
    )
    ws = [vf[:-1] for vf in vfs]
    bs = [vf[-1] for vf in vfs]
    SVals = [np.dot(w.T, DVAL) + b for w, b in zip(ws, bs)]
    LPs = [SVal > 0 for SVal in SVals]
    for i, LP in enumerate(LPs):
        error_rate = round(1 - (sum(LP == LVAL) / len(LVAL)), 4)
        lambda_val = round(log_spaced_lambda[i], 4)
        print(f"Error rate: {error_rate}, lambda: {lambda_val}")
    emp_p = sum(LTR == 1) / len(LTR)
    print(f"Empirical prior: {emp_p}")
    llrs = SVals - np.log(emp_p / (1 - emp_p))
    DCFs = [optimal_bayes_decision(llr, LVAL, pi, 1, 1)[2] for llr in llrs]
    minDCFs = [
        min(minimum_detection_cost(np.sort(llr), llr, LVAL, pi, 1, 1)[2])
        for llr in llrs
    ]
    min_minDCF = min(minDCFs)
    min_lambda = log_spaced_lambda[minDCFs.index(min_minDCF)]
    print(f"Best minDCF: {min_minDCF} for lambda: {min_lambda}")
    plot_DCF_minDCF(
        log_spaced_lambda,
        DCFs,
        minDCFs,
        saveTitle="./lab_8_weighted.png",
        title="Weighted data - DCF and MinDCF vs Lambda",
    )

    print("\nQuadratic data")
    log_spaced_lambda = np.logspace(-4, 2, 13)
    (DTR, LTR), (DVAL, LVAL) = split(quadratic_feature_expansion(data), labels)
    x0 = np.zeros(DTR.shape[0] + 1)
    pi = 0.1
    vfs, j = zip(
        *[
            (result[0], result[1])
            for result in [
                scipy.optimize.fmin_l_bfgs_b(
                    logreg_obj, x0=x0, args=(DTR, LTR, l), approx_grad=False
                )
                for l in log_spaced_lambda
            ]
        ]
    )
    ws = [vf[:-1] for vf in vfs]
    bs = [vf[-1] for vf in vfs]
    SVals = [np.dot(w.T, DVAL) + b for w, b in zip(ws, bs)]
    LPs = [SVal > 0 for SVal in SVals]
    for i, LP in enumerate(LPs):
        error_rate = round(1 - (sum(LP == LVAL) / len(LVAL)), 4)
        lambda_val = round(log_spaced_lambda[i], 4)
        print(f"Error rate: {error_rate}, lambda: {lambda_val}")
    emp_p = sum(LTR == 1) / len(LTR)
    print(f"Empirical prior: {emp_p}")
    llrs = SVals - np.log(emp_p / (1 - emp_p))
    DCFs = [optimal_bayes_decision(llr, LVAL, pi, 1, 1)[2] for llr in llrs]
    minDCFs = [
        min(minimum_detection_cost(np.sort(llr), llr, LVAL, pi, 1, 1)[2])
        for llr in llrs
    ]
    min_minDCF = min(minDCFs)
    min_lambda = log_spaced_lambda[minDCFs.index(min_minDCF)]
    best_llr = llrs[minDCFs.index(min_minDCF)]
    best_w = ws[minDCFs.index(min_minDCF)]
    best_b = bs[minDCFs.index(min_minDCF)]
    # save best llr as np array
    np.save("best_llr_logistic_regression.npy", best_llr)
    np.save("best_w_logistic_regression.npy", best_w)
    np.save("best_b_logistic_regression.npy", best_b)
    print(f"Best minDCF: {min_minDCF} for lambda: {min_lambda}")
    plot_DCF_minDCF(
        log_spaced_lambda,
        DCFs,
        minDCFs,
        saveTitle="./lab_8_quadratic.png",
        title="Quadratic data - DCF and MinDCF vs Lambda",
    )

    print("\nCentered data")
    log_spaced_lambda = np.logspace(-4, 2, 13)
    (DTR, LTR), (DVAL, LVAL) = split(data, labels)
    mu = vcol(DTR.mean(axis=1))
    DTRC = DTR - mu
    DVALC = DVAL - mu
    x0 = np.zeros(DTR.shape[0] + 1)
    pi = 0.1
    vfs, j = zip(
        *[
            (result[0], result[1])
            for result in [
                scipy.optimize.fmin_l_bfgs_b(
                    logreg_obj, x0=x0, args=(DTRC, LTR, l), approx_grad=False
                )
                for l in log_spaced_lambda
            ]
        ]
    )
    ws = [vf[:-1] for vf in vfs]
    bs = [vf[-1] for vf in vfs]
    SVals = [np.dot(w.T, DVALC) + b for w, b in zip(ws, bs)]
    LPs = [SVal > 0 for SVal in SVals]
    for i, LP in enumerate(LPs):
        error_rate = round(1 - (sum(LP == LVAL) / len(LVAL)), 4)
        lambda_val = round(log_spaced_lambda[i], 4)
        print(f"Error rate: {error_rate}, lambda: {lambda_val}")
    emp_p = sum(LTR == 1) / len(LTR)
    print(f"Empirical prior: {emp_p}")
    llrs = SVals - np.log(emp_p / (1 - emp_p))
    DCFs = [optimal_bayes_decision(llr, LVAL, pi, 1, 1)[2] for llr in llrs]
    minDCFs = [
        min(minimum_detection_cost(np.sort(llr), llr, LVAL, pi, 1, 1)[2])
        for llr in llrs
    ]
    min_minDCF = min(minDCFs)
    min_lambda = log_spaced_lambda[minDCFs.index(min_minDCF)]
    print(f"Best minDCF: {min_minDCF} for lambda: {min_lambda}")
    plot_DCF_minDCF(
        log_spaced_lambda,
        DCFs,
        minDCFs,
        saveTitle="./lab_8_centered.png",
        title="Centered data - DCF and MinDCF vs Lambda",
    )


def lab_9():
    print("\nLab 9 - Support vector machines")
    data, labels = load("./project/train.csv")
    (DTR, LTR), (DVAL, LVAL) = split(data, labels)
    log_C = np.logspace(-5, 0, 11)
    pi = 0.1
    minDCFs = []
    actDCFs = []
    for C in log_C:
        _, _, _, DCF, min_DCF = svm(DTR, LTR, DVAL, LVAL, C, 1, pi=pi)
        minDCFs.append(min_DCF)
        actDCFs.append(DCF)
    plot_DCF_minDCF(
        log_C,
        actDCFs,
        minDCFs,
        saveTitle="./lab_9.png",
        title="SVM - DCF and MinDCF vs C",
    )
    min_minDCF = min(minDCFs)
    min_C = log_C[minDCFs.index(min_minDCF)]
    print(f"Minimum minDCF: {min_minDCF} at C: {min_C}")

    print("\nCentered SVM")
    mu = vcol(DTR.mean(axis=1))
    DTRC = DTR - mu
    DVALC = DVAL - mu
    minDCFs_centered = []
    actDCFs_centered = []
    for C in log_C:
        _, _, _, DCF, min_DCF = svm(DTRC, LTR, DVALC, LVAL, C, 1, pi=pi)
        minDCFs_centered.append(min_DCF)
        actDCFs_centered.append(DCF)
    plot_DCF_minDCF(
        log_C,
        actDCFs_centered,
        minDCFs_centered,
        saveTitle="./lab_9_centered.png",
        title="SVM with Centered data - DCF and MinDCF vs C",
    )
    min_minDCF = min(minDCFs_centered)
    min_C = log_C[minDCFs_centered.index(min_minDCF)]
    print(f"Minimum minDCF: {min_minDCF} at C: {min_C}")

    print("\nKernel Polynomial SVM")
    minDCFs_kernel_poly = []
    actDCFs_kernel_poly = []
    for C in log_C:
        _, _, _, DCF, min_DCF = svm(
            DTR, LTR, DVAL, LVAL, C, 1, kernel=True, pi=pi, c=1, d=2, epsilon=0
        )
        minDCFs_kernel_poly.append(min_DCF)
        actDCFs_kernel_poly.append(DCF)
    plot_DCF_minDCF(
        log_C,
        actDCFs_kernel_poly,
        minDCFs_kernel_poly,
        saveTitle="./lab_9_kernel_poly.png",
        title="Kernel Polynomial SVM - DCF and MinDCF vs C",
    )
    min_minDCF = min(minDCFs_kernel_poly)
    min_C = log_C[minDCFs_kernel_poly.index(min_minDCF)]
    print(f"Minimum minDCF: {min_minDCF} at C: {min_C}")

    print("\nKernel RBF SVM")
    log_C = np.logspace(-3, 2, 11)
    minDCFs_kernel_rbf = {}
    actDCFs_kernel_rbf = {}
    gamma = [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]
    gammaName = ["e^-4", "e^-3", "e^-2", "e^-1"]
    best_minDCF = float("inf")
    best_llr = None
    best_C = None
    best_gamma = None
    loops = len(log_C) * len(gamma)
    n_loop = 0
    for C in log_C:
        for i, g in enumerate(gamma):
            _, _, _, DCF, min_DCF, llr = svm(
                DTR,
                LTR,
                DVAL,
                LVAL,
                C,
                1,
                kernel=True,
                kernel_type="rbf",
                pi=pi,
                epsilon=1,
                gamma=g,
                saveLLR=True,
            )
            minDCFs_kernel_rbf[gammaName[i]] = minDCFs_kernel_rbf.get(
                gammaName[i], []
            ) + [min_DCF]
            actDCFs_kernel_rbf[gammaName[i]] = actDCFs_kernel_rbf.get(
                gammaName[i], []
            ) + [DCF]
            if min_DCF < best_minDCF:
                best_minDCF = min_DCF
                best_llr = llr
                best_C = C
                best_gamma = g
            n_loop += 1
            print(f"progress: {n_loop}/{loops}")

    # plot
    plt.figure(figsize=(10, 6))
    for i, g in enumerate(gamma):
        plt.plot(
            log_C,
            actDCFs_kernel_rbf[gammaName[i]],
            label=f"Actual DCF - l:{g}",
            marker="o",
        )
        plt.plot(
            log_C,
            minDCFs_kernel_rbf[gammaName[i]],
            label=f"Minimum DCF - l:{g}",
            marker="x",
        )
    plt.xscale("log", base=10)
    plt.xlabel("C (log scale)")
    plt.ylabel("DCF Values")
    plt.title("RBF Kernel SVM - DCF and MinDCF vs C")
    plt.legend()
    plt.savefig("./lab_9_kernel_rbf.png")
    plt.clf()
    # end_plot
    print(f"Minimum minDCF: {best_minDCF} at C: {best_C} and gamma: {best_gamma}")
    np.save("best_llr_svm.npy", best_llr)
    svm(
        DTR,
        LTR,
        DVAL,
        LVAL,
        best_C,
        1,
        kernel=True,
        kernel_type="rbf",
        pi=pi,
        epsilon=1,
        gamma=best_gamma,
        saveLLR=True,
        model_filename_to_save="./best_svm.json",
    )


def lab_10():
    data, labels = load("./project/train.csv")
    (DTR, LTR), (DVAL, LVAL) = split(data, labels)

    best_numC0 = None
    best_numC1 = None
    best_cov_type = None
    best_llr = None
    best_gmm1 = None
    best_gmm0 = None
    best_minDCF = float("inf")
    num_components = [1, 2, 4, 8, 16, 32]
    cov_types = ["full", "diagonal"]
    for cov_type in cov_types:
        for numC0 in num_components:
            for numC1 in num_components:
                gmm0 = gmm_lbg_em(
                    DTR[:, LTR == 0],
                    numC0,
                    constrain_eigenvalues=True,
                    psi=0.01,
                    threshold=10**-6,
                    cov_type=cov_type,
                )
                gmm1 = gmm_lbg_em(
                    DTR[:, LTR == 1],
                    numC1,
                    constrain_eigenvalues=True,
                    psi=0.01,
                    threshold=10**-6,
                    cov_type=cov_type,
                )
                SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
                actDCF = optimal_bayes_decision(SLLR, LVAL, 0.1, 1, 1)[2]
                minDCF = min(
                    minimum_detection_cost(np.sort(SLLR), SLLR, LVAL, 0.1, 1, 1)[2]
                )
                # print(f'cov_type: {cov_type}, numC0 = {numC0}, numC1 = {numC1}: minDCF = {minDCF}, actDCF = {actDCF}')
                if minDCF < best_minDCF:
                    best_minDCF = minDCF
                    best_numC0 = numC0
                    best_numC1 = numC1
                    best_cov_type = cov_type
                    best_llr = SLLR
                    best_gmm0 = gmm0
                    best_gmm1 = gmm1
    print(
        f"Best configuration: cov_type = {best_cov_type}, numC0 = {best_numC0}, numC1 = {best_numC1} with minDCF = {best_minDCF}"
    )
    save_gmm_model(
        best_gmm0,
        best_gmm1,
        best_numC0,
        best_numC1,
        best_cov_type,
        filepath="./best_gmm.json",
    )
    np.save("best_llr_gmm.npy", best_llr)

    # plot bayeses error
    best_llr_logistic_regression = np.load("best_llr_logistic_regression.npy")
    best_llr_svm = np.load("best_llr_svm.npy")
    effPriorLogOdds = np.linspace(-4, 4, 25)
    pis = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    threshold_best_logistic = np.sort(best_llr_logistic_regression)
    threshold_best_gmm = np.sort(best_llr)
    threshold_best_svm = np.sort(best_llr_svm)
    bayes_error_plot_multiple(
        effPriorLogOdds,
        [best_llr_logistic_regression, best_llr_svm, best_llr],
        LVAL,
        [threshold_best_logistic, threshold_best_svm, threshold_best_gmm],
        pis,
        1.0,
        1.0,
        ["Logistic Regression", "SVM - RBF Kernel", "GMM"],
        title="Logistic Regression vs SVM vs GMM",
        saveTitle="./lab_10.png",
    )


def lab_11():
    data, labels = load("./project/train.csv")
    (DTR, LTR), (DVAL, LVAL) = split(data, labels)

    gmm_scores = np.load("best_llr_gmm.npy")
    svm_scores = np.load("best_llr_logistic_regression.npy")
    lr_scores = np.load("best_llr_svm.npy")

    score_sets = {"gmm": gmm_scores, "svm": svm_scores, "lr": lr_scores}
    best_results = {}

    for model_name, scores in score_sets.items():
        best_actDCF = float("inf")
        best_pi = None
        best_calibrated_scores = None
        best_labels = None

        print(f"Calibrating {model_name} scores...")

        for pi in np.linspace(0.1, 0.9, 9):
            calibrated_scores = []
            labels_combined = []

            for fold_idx in range(5):
                SCAL, SVAL = extract_train_val_folds_from_ary(scores, fold_idx)
                LCAL, LVAL_ = extract_train_val_folds_from_ary(LVAL, fold_idx)

                SCAL = vrow(SCAL)

                x0 = np.zeros(SCAL.shape[0] + 1)

                vf = scipy.optimize.fmin_l_bfgs_b(
                    logreg_obj_wei, x0=x0, args=(SCAL, LCAL, 0, pi)
                )[0]
                w = vf[:-1]
                b = vf[-1]

                calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(pi / (1 - pi))).ravel()

                calibrated_scores.append(calibrated_SVAL)
                labels_combined.append(LVAL_)

            calibrated_scores = np.hstack(calibrated_scores)
            labels_combined = np.hstack(labels_combined)

            _, _, actDCF = optimal_bayes_decision(
                calibrated_scores, labels_combined, 0.1, 1, 1
            )
            if actDCF < best_actDCF:
                best_actDCF = actDCF
                best_pi = pi
                best_calibrated_scores = calibrated_scores
                best_labels = labels_combined
        best_results[model_name] = {
            "best_actDCF": best_actDCF,
            "best_pi": best_pi,
            "best_calibrated_scores": best_calibrated_scores,
            "best_labels": best_labels,
        }
    for model_name, result in best_results.items():
        print(f"\nBest results for {model_name}:")
        print(f"  Best actDCF: {result['best_actDCF']}")
        print(f"  Best pi: {result['best_pi']}")
    for model_name, scores in score_sets.items():
        scores = vrow(scores)
        x0 = np.zeros(scores.shape[0] + 1)
        vf = scipy.optimize.fmin_l_bfgs_b(
            logreg_obj_wei,
            x0=x0,
            args=(scores, LVAL, 0, best_results[model_name]["best_pi"]),
        )[0]
        w = vf[:-1]
        b = vf[-1]
        best_results[model_name]["w"] = w
        best_results[model_name]["b"] = b

    best_fusion_actDCF = float("inf")
    best_fusion_pi = None

    for pi in np.linspace(0.1, 0.9, 9):
        calibrated_scores = []
        labels_combined = []

        for fold_idx in range(5):
            SCAL1, SVAL1 = extract_train_val_folds_from_ary(lr_scores, fold_idx)
            SCAL2, SVAL2 = extract_train_val_folds_from_ary(svm_scores, fold_idx)
            SCAL3, SVAL3 = extract_train_val_folds_from_ary(gmm_scores, fold_idx)
            LCAL, LVAL_ = extract_train_val_folds_from_ary(LVAL, fold_idx)

            SCAL = np.vstack([SCAL1, SCAL2, SCAL3])
            SVAL = np.vstack([SVAL1, SVAL2, SVAL3])

            x0 = np.zeros(SCAL.shape[0] + 1)
            vf = scipy.optimize.fmin_l_bfgs_b(
                logreg_obj_wei, x0=x0, args=(SCAL, LCAL, 0, pi)
            )[0]
            w = vf[:-1]
            b = vf[-1]

            calibrated_SVAL = (w.T @ SVAL + b - np.log(pi / (1 - pi))).ravel()

            calibrated_scores.append(calibrated_SVAL)
            labels_combined.append(LVAL_)

        calibrated_scores = np.hstack(calibrated_scores)
        labels_combined = np.hstack(labels_combined)

        _, _, actDCF = optimal_bayes_decision(
            calibrated_scores, labels_combined, 0.1, 1, 1
        )

        if actDCF < best_fusion_actDCF:
            best_fusion_actDCF = actDCF
            best_fusion_pi = pi
            best_fusion_calibrated_scores = calibrated_scores
            best_fusion_lval = labels_combined

    all_scores = np.vstack([lr_scores, svm_scores, gmm_scores])
    x0 = np.zeros(SCAL.shape[0] + 1)
    vf = scipy.optimize.fmin_l_bfgs_b(
        logreg_obj_wei, x0=x0, args=(all_scores, LVAL, 0, best_fusion_pi)
    )[0]
    fusion_w = vf[:-1]
    fusion_b = vf[-1]
    effPriorLogOdds = np.linspace(-3, 3, 21)
    pis = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    threshold_best_logistic = np.sort(best_results["lr"]["best_calibrated_scores"])
    threshold_best_gmm = np.sort(best_results["gmm"]["best_calibrated_scores"])
    threshold_fusion = np.sort(best_fusion_calibrated_scores)
    threshold_best_svm = np.sort(best_results["svm"]["best_calibrated_scores"])
    bayes_error_plot_multiple(
        effPriorLogOdds,
        [
            best_results["lr"]["best_calibrated_scores"],
            best_results["svm"]["best_calibrated_scores"],
            best_results["gmm"]["best_calibrated_scores"],
            best_fusion_calibrated_scores,
        ],
        [
            best_results["lr"]["best_labels"],
            best_results["svm"]["best_labels"],
            best_results["gmm"]["best_labels"],
            best_fusion_lval,
        ],
        [
            threshold_best_logistic,
            threshold_best_svm,
            threshold_best_gmm,
            threshold_fusion,
        ],
        pis,
        1.0,
        1.0,
        ["Logistic Regression", "SVM - RBF Kernel", "GMM", "Fusion"],
        left=-3,
        right=3,
        multipleLabels=True,
        title="Logistic Regression vs SVM vs GMM vs Fusion calibrations",
        saveTitle="./lab_11.png",
    )
    best_actDCF = float("inf")
    best_pi = None
    best_calibrated_scores = None
    best_model = None
    for model_name, result in best_results.items():
        if result["best_actDCF"] < best_actDCF:
            best_actDCF = result["best_actDCF"]
            best_pi = result["best_pi"]
            best_calibrated_scores = result["best_calibrated_scores"]
            best_lval = result["best_labels"]
            best_model = model_name
    if best_fusion_actDCF < best_actDCF:
        best_actDCF = best_fusion_actDCF
        best_pi = best_fusion_pi
        best_calibrated_scores = best_fusion_calibrated_scores
        best_lval = best_fusion_lval
        best_model = "Fusion"
    print(f"Best model: {best_model}, best actDCF: {best_actDCF}, best pi: {best_pi}")

    test_set_data, test_set_label = load("./project/evalData.txt")
    tsd_ext = quadratic_feature_expansion(test_set_data)

    w__ = np.load("best_w_logistic_regression.npy")
    b__ = np.load("best_b_logistic_regression.npy")
    llr_eval_lr = np.dot(w__.T, tsd_ext) + b__ - np.log(pi / (1 - pi))
    llr_eval_lr_calibrated = (
        best_results["lr"]["w"].T @ vrow(llr_eval_lr)
        + best_results["lr"]["b"]
        - np.log(best_results["lr"]["best_pi"] / (1 - best_results["lr"]["best_pi"]))
    ).ravel()
    llr_eval_svm = compute_llr_rbf(load_svm_model_json("best_svm.json"), test_set_data)
    llr_eval_svm_calibrated = (
        best_results["svm"]["w"].T @ vrow(llr_eval_svm)
        + best_results["svm"]["b"]
        - np.log(best_results["svm"]["best_pi"] / (1 - best_results["svm"]["best_pi"]))
    ).ravel()
    gmm0, gmm1, _, _, _ = load_gmm_model("./best_gmm.json")
    llr_eval_gmm = compute_llr_for_new_data_gmm(gmm0, gmm1, test_set_data)
    llr_eval_gmm_calibrated = (
        best_results["gmm"]["w"].T @ vrow(llr_eval_gmm)
        + best_results["gmm"]["b"]
        - np.log(best_results["gmm"]["best_pi"] / (1 - best_results["gmm"]["best_pi"]))
    ).ravel()
    scores_eval = np.vstack([llr_eval_lr, llr_eval_svm, llr_eval_gmm])
    scores_eval_calibrated = (
        fusion_w.T @ scores_eval
        + fusion_b
        - np.log(best_fusion_pi / (1 - best_fusion_pi))
    ).ravel()
    effPriorLogOdds = np.linspace(-3, 3, 21)

    pis = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    threshold_cal_logistic = np.sort(llr_eval_lr_calibrated)
    threshold_cal_gmm = np.sort(llr_eval_gmm_calibrated)
    threshold_cal_fusion = np.sort(scores_eval_calibrated)
    threshold_cal_svm = np.sort(llr_eval_svm_calibrated)
    bayes_error_plot_multiple(
        effPriorLogOdds,
        [llr_eval_svm_calibrated, llr_eval_gmm_calibrated, scores_eval_calibrated],
        # [llr_eval_lr_calibrated, llr_eval_svm_calibrated, llr_eval_gmm_calibrated, scores_eval_calibrated],
        test_set_label,
        [threshold_cal_svm, threshold_cal_gmm, threshold_cal_fusion],
        # [threshold_cal_logistic, threshold_cal_svm, threshold_cal_gmm, threshold_cal_fusion],
        pis,
        1.0,
        1.0,
        ["SVM - RBF Kernel", "GMM", "Fusion"],
        # ["Logistic Regression", "SVM - RBF Kernel", "GMM", "Fusion"],
        left=-3,
        right=3,
        title="SVM vs GMM vs Fusion - DCF vs Prior Log Odds on Eval Set",
        saveTitle="./lab_11_model_comparison.png",
        multipleLabels=False,
    )
    bayes_error_plot(
        effPriorLogOdds,
        scores_eval_calibrated,
        test_set_label,
        threshold_cal_fusion,
        pis,
        1,
        1,
        left=-3,
        right=3,
        saveTitle="./lab_11_delivered_model.png",
        title="Fusion - DCF vs Prior Log Odds",
    )


run_all_lab = False
lab_to_run = 11
if run_all_lab:
    lab_2()
    lab_3()
    lab_4()
    lab_5()
    lab_7()
    lab_8()
    lab_9()
    lab_10()
    lab_11()
else:
    if lab_to_run == 2:
        lab_2()
    elif lab_to_run == 3:
        lab_3()
    elif lab_to_run == 4:
        lab_4()
    elif lab_to_run == 5:
        lab_5()
    elif lab_to_run == 7:
        lab_7()
    elif lab_to_run == 8:
        lab_8()
    elif lab_to_run == 9:
        lab_9()
    elif lab_to_run == 10:
        lab_10()
    elif lab_to_run == 11:
        lab_11()
