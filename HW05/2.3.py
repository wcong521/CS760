import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

def BuggyPCA(X):
    plt.subplot(2, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='.', color="white", label="Original")

    U, S, VT = np.linalg.svd(X)
    B = VT.T[:, 0]
    Z = X @ B
    X_tilde = np.outer(Z, B.T)

    plt.scatter(X_tilde[:, 0], X_tilde[:, 1], marker='.', color="orange", label="Reconstructed")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.legend()
    plt.title("Buggy PCA")

    re = np.linalg.norm(X - X_tilde, ord=2) ** 2
    print(re)


def DemeanedPCA(X):

    plt.subplot(2, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], marker='.', color="white", label="Original")

    mean = np.mean(X, axis=0)

    X_centered = X - mean

    U, S, VT = np.linalg.svd(X_centered)
    B = VT.T[:, 0]
    Z = X_centered @ B
    X_tilde = np.outer(Z, B.T) + mean

    plt.scatter(X_tilde[:, 0], X_tilde[:, 1], marker='.', color="orange", label="Reconstructed")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.legend()
    plt.title("Demeaned PCA")

    re = np.linalg.norm(X - X_tilde, ord=2) ** 2
    print(re)


def NormalizedPCA(X):

    plt.subplot(2, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], marker='.', color="white", label="Original")

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    U, S, VT = np.linalg.svd(X_normalized)
    B = VT.T[:, 0]
    Z = X_normalized @ B
    X_tilde = std * np.outer(Z, B.T) + mean


    plt.scatter(X_tilde[:, 0], X_tilde[:, 1], marker='.', color="orange", label="Reconstructed")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.legend()
    plt.title("Normalized PCA")

    re = np.linalg.norm(X - X_tilde, ord=2) ** 2
    print(re)

def DRO(X):
    plt.subplot(2, 3, 4)
    plt.scatter(X[:, 0], X[:, 1], marker='.', color="white", label="Original")

    d = 1

    mean = np.mean(X, axis=0)
    # std = np.std(X, axis=0)

    X_centered = X - mean
    U, S, VT = np.linalg.svd(X_centered)
    n = X_centered.shape[0]

    Z = np.sqrt(n) * U[:, :d]
    A = 1/np.sqrt(n) * VT[:d].T @ np.diag(S[:d])

    X_tilde = Z @ A.T + mean

    plt.scatter(X_tilde[:, 0], X_tilde[:, 1], marker='.', color="orange", label="Reconstructed")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.legend()
    plt.title("DRO")

    re = np.linalg.norm(X - X_tilde, ord=2) ** 2
    print(re)


X2 = np.genfromtxt('./data/data2D.csv', delimiter=',', dtype=float)
X1000 = np.genfromtxt('./data/data1000D.csv', delimiter=',', dtype=float)


plt.figure(figsize=(10, 8))

U, S, VT = np.linalg.svd(X1000)
plt.subplot(2, 3, 5)
plt.plot(S)
plt.yscale("log")
plt.title("DRO Singular Values")

BuggyPCA(np.copy(X2))
DemeanedPCA(np.copy(X2))
NormalizedPCA(np.copy(X2))
DRO(np.copy(X2))

plt.show()
