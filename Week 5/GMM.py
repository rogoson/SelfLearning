import numpy as np
import scipy.stats
from scipy.stats import multivariate_normal, halfnorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def initialiseMixtureWeights(n=4):
    weights = np.random.rand(n)
    return weights / np.sum(weights)


def initialiseMeans(dim=2, n=4):
    means = np.random.normal(0, scale=4, size=(n, dim))
    return means


def getCovariances(n=4, std=3):
    matrices = []
    for i in range(n):
        matrices.append(
            np.array(
                [
                    [halfnorm.rvs(loc=0, scale=std), 0],
                    [0, halfnorm.rvs(loc=0, scale=std)],
                ]
            )
        )
    return matrices


def generateData(n=200, clusters=4):
    means = [[0, 2], [2, 2], [0, 0], [2, 0]]
    covariances = getCovariances(n=clusters)
    sampledPoints = []
    for _ in range(clusters):
        currMean = means.pop()
        currCovariance = covariances.pop()
        sampledPoints.append(
            multivariate_normal(currMean, cov=currCovariance).rvs(size=n // clusters)
        )
    return np.vstack(sampledPoints)


points = generateData()

initMixWeights = initialiseMixtureWeights()
initMeans = initialiseMeans()
covariances = getCovariances()


x, y = points[:, 0], points[:, 1]


def computeResponsibilities(weights, means, covariances, points):
    responsibilityMatrix = np.zeros((means.shape[0], points.shape[0]))
    for m in range(len(means)):
        for p in range(len(points)):
            responsibilityMatrix[m][p] = weights[
                m
            ] * scipy.stats.multivariate_normal.pdf(points[p], means[m], covariances[m])
    responsibilityMatrix /= responsibilityMatrix.sum(axis=0)
    return responsibilityMatrix


def maximisation(responsibiltyMat, points):
    effNumber = np.sum(responsibiltyMat, axis=1)
    # for mean in mean
    transposedResponsib = responsibiltyMat.T
    newMeans = (points.T @ transposedResponsib).T
    newMeans /= effNumber[:, np.newaxis]

    newCovariances = []
    for cluster in range(len(newMeans)):
        newCov = (
            (
                responsibiltyMat[cluster, :][:, np.newaxis]
                * (points - newMeans[cluster])
            ).T
            @ (points - newMeans[cluster])
        ) / effNumber[cluster]
        newCov += 1e-6 * np.eye(newCov.shape[0])
        newCovariances.append(newCov)

    newWeights = effNumber / len(points)
    return newWeights, newMeans, np.array(newCovariances)


means = [initMeans.copy()]

for iteration in range(15):
    respMatrix = computeResponsibilities(initMixWeights, initMeans, covariances, points)
    initMixWeights, initMeans, covariances = maximisation(respMatrix, points)
    means.append(initMeans)

## AI write the below


def draw_ellipse(mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellip = patches.Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        fc="None",
        lw=2,
    )
    plt.gca().add_patch(ellip)


def plot_gmm(points, means, covariances, responsibilities):
    points = np.array(points)
    means = np.array(means)
    K = means.shape[1]
    colors = ["red", "green", "orange", "purple"]

    # Use final responsibilities to assign cluster
    assignments = np.argmax(responsibilities, axis=0)

    for k in range(K):
        cluster_pts = points[assignments == k]
        plt.scatter(
            cluster_pts[:, 0], cluster_pts[:, 1], color=colors[k], alpha=0.5, s=10
        )

        # Plot trajectory of mean
        plt.plot(
            means[:, k, 0], means[:, k, 1], color=colors[k], linestyle="--", linewidth=1
        )

        # Final mean
        plt.scatter(
            means[-1, k, 0], means[-1, k, 1], color=colors[k], s=100, marker="x"
        )

        # Covariance ellipse
        draw_ellipse(means[-1, k], covariances[k], color=colors[k])

    plt.title("GMM Clusters and Mean Trajectories")
    plt.axis("equal")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(5)
    plt.close()


plot_gmm(points, means, covariances, respMatrix)
