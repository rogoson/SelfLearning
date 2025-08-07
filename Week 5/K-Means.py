import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, halfnorm
from collections import defaultdict


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

x, y = points[:, 0], points[:, 1]
plt.scatter(x, y, color="blue")


def initialiseCentroids(points, num=4):
    # randomly choose num points from len(points) number of indices - gives indices, and then
    idx = np.random.choice(len(points), num, replace=False)
    return points[idx]


centroids = initialiseCentroids(points)
for cent in range(centroids.shape[0]):
    x, y = centroids[cent, 0], centroids[cent, 1]
    plt.scatter(x, y, label=f"Centroid: {cent+1}")
plt.legend()
plt.show(block=False)
plt.pause(5)
plt.close()


def optimise(points, centroids, optStep=0):
    assignmentClusters = defaultdict(list)
    for pointNumber in range(len(points)):
        currMin = np.inf
        bestCluster = -1
        for centroidNumber in range(len(centroids)):
            euclideanDist = np.linalg.norm(
                points[pointNumber] - centroids[centroidNumber]
            )
            if euclideanDist < currMin:
                bestCluster = centroidNumber
                currMin = euclideanDist
        assignmentClusters[f"{bestCluster}"].append(points[pointNumber])

    allNewPoints = []
    allNewMeans = []
    for ind in range(len(centroids)):
        if len(assignmentClusters[f"{ind}"]) == 0:
            newMean = points[np.random.randint(len(points))]
        else:
            allPoints = np.array(assignmentClusters[f"{ind}"])
            newMean = np.mean(allPoints, axis=0)
            allNewPoints.append(allPoints)

        allNewMeans.append(newMean)
        x, y = allPoints[:, 0], allPoints[:, 1]
        plt.scatter(newMean[0], newMean[1], label=f"Centroid: {ind+1}")
        plt.scatter(x, y, label=f"Centroid {ind + 1} Points")
    plt.legend()
    plt.show(block=False)
    plt.title(f"Optimisation Step (K-Means): {optStep}")
    plt.pause(5)
    plt.close()

    return np.vstack(allNewPoints), allNewMeans


firstInd = 0
while True:
    pts, cts = optimise(
        points=points.reshape(-1, 2), centroids=centroids, optStep=firstInd + 1
    )
    firstInd += 1
    if np.allclose(centroids, cts):
        break
    points, centroids = pts, cts
    # add some stopping condition when no more class moving but good enough ig
