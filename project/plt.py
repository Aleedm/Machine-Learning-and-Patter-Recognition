import numpy as np
import matplotlib.pyplot as plt


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