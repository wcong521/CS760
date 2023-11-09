import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components, n=100):
        self.n_components = n_components
        self.n = n

    def fit(self, X):
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = [np.identity(n_features) for _ in range(self.n_components)]

        for i in range(self.n):
            r = self.e_step(X)
            self.m_step(X, r)
    
        r = self.e_step(X)
        return np.argmax(r, axis=1), self.means, self.covariances, self.compute_loss(X)

    def e_step(self, X):
        r = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            # print(multivariate_normal.pdf(X, self.means[i], self.covariances[i]))
            r[:, i] = self.weights[i] * multivariate_normal.pdf(X, self.means[i], self.covariances[i])

        r /= r.sum(axis=1, keepdims=True)
        return r

    def m_step(self, X, r):
        for i in range(self.n_components):
            total_r = r[:, i].sum()
            self.weights[i] = total_r / X.shape[0]
            self.means[i] = np.sum(X * r[:, i][:, np.newaxis], axis=0) / total_r
            self.covariances[i] = np.dot((X - self.means[i]).T, (X - self.means[i]) * r[:, i][:, np.newaxis]) / total_r

    def compute_loss(self, X):
        log_likelihood = 0
        for x in X:
            likelihood = 0
            for i in range(self.n_components):
                likelihood += self.weights[i] * multivariate_normal.pdf(x, self.means[i], self.covariances[i])
            log_likelihood += np.log(likelihood)
        return -log_likelihood