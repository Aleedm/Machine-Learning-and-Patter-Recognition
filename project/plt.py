import numpy as np
import matplotlib.pyplot as plt
from project.utils import vrow, cov
from project.gaussian_density_estimation import logpdf_GAU_ND, loglikelihood


def plot_hist(feature, L, bins=10, threshold=None):
    l = ["False", "True"]
    plt.hist(feature[L == 0].flatten(), alpha=0.5, label=l[0], bins=bins, density=True)
    plt.hist(feature[L == 1].flatten(), alpha=0.5, label=l[1], bins=bins, density=True)
    if threshold is not None:
        plt.axvline(x=threshold, color="green", linestyle="--", label="Threshold")
    plt.legend()
    plt.show()


def plot_scatter(feat1, feat1_name, feat2, feat2_name, L):
    l = ["False", "True"]
    plt.scatter(feat1[L == 0], feat2[L == 0], label=l[0])
    plt.scatter(feat1[L == 1], feat2[L == 1], label=l[1])
    plt.xlabel(feat1_name)
    plt.ylabel(feat2_name)
    plt.legend()
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


def plot_gau_den_est(D_t, D_f, project = None):
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
    #res = logpdf_GAU_ND(vrow(D), mu, C)
    #max_d, min_d = np.max(D), np.min(D)
    #XPlot = np.linspace(min_d, max_d, D.shape[0])
    #plt.plot(XPlot.ravel(), np.exp(res))
    if(project is not None):
        print(f"likelihood of class {project[0]} for feature {project[1]}: {loglikelihood(vrow(D_t), mu_t, C_t)}")
