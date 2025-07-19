import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

MAX_X = 20

"""
When to use primal form:
- Many points, few features so you can have a smaller linear model and it's not that 
hard to fit the regression function
When to use dual form:
- fewer points, many features where inverting the N x N matrix is more feasible
and computing the phi(x) isn't really practical for all the ws
"""


def generatePoints(noise=0.2, predictions=None):
    xValues = np.linspace(0, MAX_X, 100)
    yValues = np.cos(xValues)
    noiseVector = np.random.normal(scale=noise, size=yValues.shape)
    points = yValues + noiseVector
    return {
        "points": points,
        "xValues": xValues,
        "yValues": yValues,
    }


def kernelFunction(point1, point2, std):
    # Gaussian Difference kernel (idk the proper name)
    return np.exp(-((point1 - point2) ** 2) / (2 * std**2))


def generateKernelMatrix(points, std=0.5):
    pointsMatrix = np.empty((0, len(points)))
    for i in range(len(points)):
        pointsRow = np.zeros(len(points))
        for j in range(len(points)):
            pointsRow[j] = kernelFunction(points[i], points[j], std)
        pointsMatrix = np.vstack([pointsMatrix, pointsRow])
    return np.array(pointsMatrix)


def getAlphaSolution(targets, kernelMatrix, lam=10):
    alphaSolution = (
        np.linalg.inv(kernelMatrix + lam * np.eye(kernelMatrix.shape[0])) @ targets
    )
    return alphaSolution


def getPredictions(predictionPointIndex, alpha, kernelMatrix):
    kernelColumn = kernelMatrix[predictionPointIndex, :]
    scalarPred = np.dot(alpha, kernelColumn)
    return scalarPred


axPosition = 0
hyperparameters = {"lambdas": [5, 10, 15, 20], "stds": [0.2, 0.5, 1, 2]}
fig, axArray = plt.subplots(
    len(hyperparameters.keys()),
    len(next(iter(hyperparameters.values()))),
    figsize=(12, 10),
)

for hyperparam, param in hyperparameters.items():
    for parameter in param:
        ax = axArray.ravel()[axPosition]
        data = generatePoints()
        xValues = data["xValues"]
        targets = data["points"]
        kernelMatrix = generateKernelMatrix(
            xValues, std=parameter if hyperparam == "stds" else 0.2
        )
        alphaSol = getAlphaSolution(
            targets,
            kernelMatrix,
            lam=parameter if hyperparam == "lambdas" else 10,
        )

        predictions = []
        for p in range(len(xValues)):
            predictions.append(getPredictions(p, alphaSol, kernelMatrix))

        mSError = np.sum((np.array(predictions) - targets) ** 2) / len(targets)
        ax.scatter(xValues, targets, label="Original Function")
        ax.plot(xValues, predictions, label="Kernel Predictions", color="orange")
        ax.set_title(f"Kernel | MSE {round(mSError, 3)} | {hyperparam}, {parameter}")
        ax.legend()
        axPosition += 1
plt.tight_layout()
plt.show(block=False)
plt.pause(10)
plt.close()


## plot heatmap of kernel matrix
# diagonal horizontal line suggests points are closely related to points that they are nearby or similarly evaluate to, but obvs as distanec increases they don't have similar evalutations
data = generatePoints()
xValues = data["xValues"]
kernelMatrix = generateKernelMatrix(xValues)

plt.figure(figsize=(10, 8))
sns.heatmap(kernelMatrix, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Kernel Matrix Evaluation")
plt.tight_layout()
plt.show(block=False)
plt.pause(10)
plt.close()
