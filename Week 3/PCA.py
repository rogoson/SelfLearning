import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()


fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
ax.set_title("Original Data")

print(iris.data.shape)
# mean-center
dataMatrix = iris.data - iris.data.mean(axis=0)
# compute covar (divide by number of datapoints to scale)
covarianceMatrix = (1 / dataMatrix.shape[0]) * (dataMatrix.T @ dataMatrix)

eigenValues, eigenVectors = np.linalg.eigh(covarianceMatrix)
print("Eigenvectors", np.matrix(eigenVectors))
print("Eigenvalues", eigenValues)


# Sort eigenvalues and eigenvectors in descending order
sortedIndices = np.argsort(eigenValues)[::-1]
eigenValues = eigenValues[sortedIndices]
eigenVectors = eigenVectors[:, sortedIndices]

ax4.bar([f"Eigenvalue {i}" for i in range(iris.data.shape[1])], eigenValues)
ax4.set_title(f"Eigenvalues")

# Choose top k eigenvectors
k = 2
projectionMatrix = eigenVectors[:, :k]  # shape (4, 2)

# project and reconstruct
projectedData = dataMatrix @ projectionMatrix  # shape (150, 2)
almostX = projectedData @ projectionMatrix.T  # shape (150, 4)


featureSquaredErrors = np.mean((almostX - dataMatrix) ** 2, axis=0)
# print squared errors for feature approximation
ax2.bar([f"Feature {i}" for i in range(iris.data.shape[1])], featureSquaredErrors)
ax2.set_title(f"Feature Errors with Approx")

scatter = ax3.scatter(almostX[:, 0], almostX[:, 1], c=iris.target)
ax3.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax3.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
ax3.set_title("PCA Data Approximation (mean centered)")
plt.show()
