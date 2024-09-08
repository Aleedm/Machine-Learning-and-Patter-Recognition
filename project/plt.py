import numpy as np
import matplotlib.pyplot as plt
from project.utils import vrow, cov
from project.gaussian_density_estimation import logpdf_GAU_ND, loglikelihood
from project.bayes_decision_model import optimal_bayes_decision, minimum_detection_cost


def plot_hist(feature, L, bins=10, threshold=None, title=None, saveTitle=None):
    l = ["False", "True"]
    plt.hist(feature[L == 0].flatten(), alpha=0.5, label=l[0], bins=bins, density=True)
    plt.hist(feature[L == 1].flatten(), alpha=0.5, label=l[1], bins=bins, density=True)
    if threshold is not None:
        plt.axvline(x=threshold, color="green", linestyle="--", label="Threshold")
    plt.legend()
    if title is not None:
        plt.title(title)
    if saveTitle is not None:
        plt.savefig(saveTitle)
        plt.clf()
    else:
        plt.show()


def plot_scatter(feat1, feat1_name, feat2, feat2_name, L, saveTitle=None):
    l = ["False", "True"]
    plt.scatter(feat1[L == 0], feat2[L == 0], label=l[0], alpha=0.5)
    plt.scatter(feat1[L == 1], feat2[L == 1], label=l[1], alpha=0.5)
    plt.xlabel(feat1_name)
    plt.ylabel(feat2_name)
    plt.legend()
    if saveTitle is not None:
        plt.savefig(saveTitle)
        plt.clf()
    else:
        plt.show()


def plot_binary_classification_results(D, threshold, L):
    class_0_indices = L == 0
    class_1_indices = L == 1

    plt.figure(figsize=(10, 2))
    plt.scatter(
        D[class_0_indices], [0] * np.sum(class_0_indices), alpha=0.5, label="False"
    )
    plt.scatter(
        D[class_1_indices], [0] * np.sum(class_1_indices), alpha=0.5, label="True"
    )

    plt.axvline(x=threshold, color="green", linestyle="--", label="Threshold")

    plt.yticks([])
    plt.xlabel("X")
    plt.title("Binary classification")
    plt.legend(loc="lower right")

    plt.show()


def plot_gau_den_est(D_t, D_f, project=None, saveTitle=None, title=None):
    plt.figure()
    plt.hist(D_t.ravel(), bins=25, density=True, alpha=0.5, color="blue", label="True")
    C_t, mu_t = cov(vrow(D_t), mu=True)
    XPlot = np.linspace(-4, 4, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu_t, C_t)), color="blue")

    plt.hist(D_f.ravel(), bins=25, density=True, alpha=0.5, color="red", label="False")
    C_f, mu_f = cov(vrow(D_f), mu=True)
    XPlot = np.linspace(-4, 4, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu_f, C_f)), color="red")
    plt.legend()
    # res = logpdf_GAU_ND(vrow(D), mu, C)
    # max_d, min_d = np.max(D), np.min(D)
    # XPlot = np.linspace(min_d, max_d, D.shape[0])
    # plt.plot(XPlot.ravel(), np.exp(res))
    if project is not None:
        print(
            f"likelihood of class {project[0]} for feature {project[1]}: {loglikelihood(vrow(D_t), mu_t, C_t)}"
        )
    if title:
        plt.title(title)
    if saveTitle:
        plt.savefig(saveTitle)
        plt.clf()


def bayes_error_plot(
    effPriorLogOdds,
    llr,
    labels,
    threshold,
    pis,
    Cfn,
    Cfp,
    title=None,
    saveTitle=None,
    left=-4,
    right=4,
):
    dcf = []
    min_dcf = []
    for pi in pis:
        _, _, dcf_ = optimal_bayes_decision(llr, labels, pi, Cfn, Cfp)
        _, _, dcfs = minimum_detection_cost(threshold, llr, labels, pi, Cfn, Cfp)
        dcf.append(dcf_)
        min_dcf.append(min(dcfs))
    plt.plot(effPriorLogOdds, dcf, label="DCF", linestyle="-")
    plt.plot(effPriorLogOdds, min_dcf, label="min DCF", linestyle="--")
    plt.xlim([left, right])
    plt.xlabel("log odds")
    plt.ylabel("DFC Value")
    plt.legend()
    if title:
        plt.title(title)
    if saveTitle:
        plt.savefig(saveTitle)
        plt.clf()
    else:
        plt.show()


def plot_DCF_minDCF(
    log_spaced_lambda, DCFs, minDCFs, title="DCF and MinDCF vs Lambda", saveTitle=None
):
    plt.figure(figsize=(10, 6))
    plt.plot(log_spaced_lambda, DCFs, label="Actual DCF", marker="o")
    plt.plot(log_spaced_lambda, minDCFs, label="Minimum DCF", marker="x")

    plt.xscale("log", base=10)

    plt.xlabel("Lambda (log scale)")
    plt.ylabel("DCF Values")
    plt.title(title)
    plt.legend()
    if saveTitle:
        plt.savefig(saveTitle)
        plt.clf()
    else:
        plt.show()


def bayes_error_plot_multiple(
    effPriorLogOdds,
    llr_list,
    label,
    threshold_list,
    pis,
    Cfn,
    Cfp,
    model_names,
    title=None,
    saveTitle=None,
    left=-4,
    right=4,
    multipleLabels=False,
):
    plt.figure(figsize=(10, 6))

    if multipleLabels == False:
        #make an array of labels for each model
        labels = [label for i in range(len(llr_list))]
    else:
        labels = label
    
    for llr, threshold, model_name, l in zip(llr_list, threshold_list, model_names, labels):
        dcf = []
        min_dcf = []

        for pi in pis:
            _, _, dcf_ = optimal_bayes_decision(llr, l, pi, Cfn, Cfp)
            _, _, dcfs = minimum_detection_cost(threshold, llr, l, pi, Cfn, Cfp)
            dcf.append(dcf_)
            min_dcf.append(min(dcfs))

        plt.plot(effPriorLogOdds, dcf, label=f"{model_name} DCF", linestyle="-")
        plt.plot(
            effPriorLogOdds, min_dcf, label=f"{model_name} min DCF", linestyle="--"
        )

    #plt.ylim([0, 1.1])
    plt.xlim([left, right])
    plt.xlabel("log odds")
    plt.ylabel("DFC Value")
    plt.legend()

    if title:
        plt.title(title)
    if saveTitle:
        plt.savefig(saveTitle)
        plt.clf()
    else:
        plt.show()
