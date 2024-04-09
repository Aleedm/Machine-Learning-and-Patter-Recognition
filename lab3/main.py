from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def load_iris():
    return datasets.load_iris()["data"].T, datasets.load_iris()["target"]


def vcol(v):
    return v.reshape(-1, 1)


def vrow(v):
    return v.reshape(1, -1)


def PCA(D, n):
    mu = D.mean(axis=1)
    DC = D - vcol(mu)
    C = np.dot(DC, DC.T) / (D.shape[0])
    _, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:n]
    DP = np.dot(P.T, D)
    return DP


def plot_iris(result, target):
    plt.scatter(result[0, target == 0], result[1, target == 0], label="Setosa")
    plt.scatter(result[0, target == 1], result[1, target == 1], label="Versicolor")
    plt.scatter(result[0, target == 2], result[1, target == 2], label="Virginica")
    plt.legend()
    plt.show()


def main():
    D, target = load_iris()
    result = PCA(D, 2)
    plot_iris(result, target)


if __name__ == "__main__":
    main()
