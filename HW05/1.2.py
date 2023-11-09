
import matplotlib.pyplot as plt
import numpy as np
from KMeans import KMeans
from GaussianMixtureModel import GaussianMixtureModel
import math
import itertools


def match_centers(target_centers, pred_centers):
    index_pairs = list(itertools.product([0, 1, 2], [0, 1, 2]))
    distances = [np.linalg.norm(target_centers[i] - pred_centers[j]) for i, j in index_pairs]

    combined_lists = list(zip(index_pairs, distances))
    sorted_combined = sorted(combined_lists, key=lambda x: x[1])
    
    target_used = set()
    pred_used = set()
    output = []
    for e in sorted_combined:
        if e[0][0] not in target_used and e[0][1] not in pred_used:
            output.append(e[0])
            target_used.add(e[0][0])
            pred_used.add(e[0][1])

    return [c[1] for c in sorted(output, key=lambda x: x[0])]

def eval_kmeans(X, target_means, target_clusters):
    kmeans_model = KMeans(3)
    min_loss = math.inf
    accuracy = None
    
    for i in range(N_RESTARTS):
        clusters, centroids, loss = kmeans_model.fit(X)

        link = match_centers(target_means, centroids)
        n_correct = 0
        for j in range(len(target_clusters)):
            if target_clusters[j] == link[clusters[j]]:
                n_correct += 1
        
        temp_accuracy = n_correct / len(target_clusters)

        if loss < min_loss:
            min_loss = loss
            accuracy = temp_accuracy

    kmeans_loss.append(min_loss)
    kmeans_accuracy.append(accuracy)

def eval_gmm(X, target_means, target_clusters):
    gmm_model = GaussianMixtureModel(3)
    min_loss = math.inf
    accuracy = None
    for i in range(N_RESTARTS):
        clusters, means, covariances, loss = gmm_model.fit(X)

        link = match_centers(target_means, means)
        n_correct = 0
        for j in range(len(target_clusters)):
            if target_clusters[j] == link[clusters[j]]:
                n_correct += 1
        
        temp_accuracy = n_correct / len(target_clusters)

        if loss < min_loss:
            min_loss = loss
            accuracy = temp_accuracy

    gmm_loss.append(min_loss)
    gmm_accuracy.append(accuracy)

plt.style.use('dark_background')
# np.random.seed(0)

kmeans_loss = []
kmeans_accuracy = []

gmm_loss = []
gmm_accuracy = []

sigmas = [0.5, 1, 2, 4, 8]

N_RESTARTS = 20

for sigma in sigmas:

    target_means = [
        np.array([-1, -1]),
        np.array([1, -1]),
        np.array([0, 1])
    ]

    target_clusters = np.repeat(np.arange(3), 100)

    A = np.random.multivariate_normal(
        target_means[0], 
        sigma * np.array([[2, 0.5], [0.5, 1]]), 
        100
    )

    B = np.random.multivariate_normal(
        target_means[1], 
        sigma * np.array([[1, -0.5], [-0.5, 2]]), 
        100
    )

    C = np.random.multivariate_normal(
        target_means[2], 
        sigma * np.array([[1, 0], [0, 2]]), 
        100
    )

    X = np.vstack((A, B, C))

    # K-Means

    eval_kmeans(X, target_means, target_clusters)

    # GMM

    eval_gmm(X, target_means, target_clusters)
    
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(sigmas, kmeans_loss, marker='o', color="orange")
plt.xticks(sigmas)
plt.xlabel(r"$\sigma$")
plt.ylabel("Objective Value")
plt.title("K-Means")

plt.subplot(2, 2, 2)
plt.plot(sigmas, gmm_loss, marker='o', color="orange")
plt.xticks(sigmas)
plt.xlabel(r"$\sigma$")
plt.ylabel("Negative Log-Likelihood")
plt.title("Gaussian Mixture Models") 

plt.subplot(2, 2, 3)
plt.plot(sigmas, kmeans_accuracy, marker='o', color="orange")
plt.xticks(sigmas)
plt.xlabel(r"$\sigma$")
plt.ylabel("Accuracy")
plt.title("K-Means")

plt.subplot(2, 2, 4)
plt.plot(sigmas, gmm_accuracy, marker='o', color="orange")
plt.xticks(sigmas)
plt.xlabel(r"$\sigma$")
plt.ylabel("Accuracy")
plt.title("Gaussian Mixture Models") 

# plt.plot(sigmas, gmm_loss, marker='o', color="orange")
# # for i in range(len(sigmas)):
# #     plt.annotate("{:.2f}".format(kmeans_loss[i]), (sigmas[i], kmeans_loss[i] + 100), fontsize=8) 
# plt.xticks(sigmas)
# plt.xlabel(r"$\sigma$")
# plt.ylabel("Objective Value")
# plt.title("K-Means")
# plt.show()    

plt.show()