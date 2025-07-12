import numpy as np
import matplotlib.pyplot as plt

"""
WE DON'T JUDGE CODE QUALITY
"""

ANALYTIC_REGRESSION = True
USE_GAUSSIAN_RBFS = True
DEGREES = 50
LR = 1e-4
BETA_1 = 0.9
BETA_2 = 0.999
MAX_X = 10


def gaussianFunctions(x, maximumX=MAX_X, degree=DEGREES): # not really accurate to say degree for rbf
    means = np.linspace(0, maximumX, degree) # 10rbfs
    stdev = 0.5
    bias = 1
    return np.array([bias] + [gaussian(x, mean, stdev) for mean in means])


def gaussian(x, mu, std):
    coefficient = 1 / (std * np.sqrt(2 * np.pi))
    exponent = np.exp((-1 / 2) * ((x - mu)/std) ** 2)
    return coefficient * exponent

def polynomial(sample, degree):
    return np.array([sample ** i for i in range(degree + 1)])


def generateBasisMatrix(X, degree, gaussianRBF = False):
    """
    Presuming only one feature:
    X: vector of
    gaussianRBF: whether to use gaussianRBFs or not (otherwise quadratic)
    """
    if gaussianRBF:
        func = gaussianFunctions
    else:
        func = polynomial

    matrix = np.empty((0, degree + 1))
    for sample in X:
        row = func(sample, degree=degree)
        matrix = np.vstack([matrix, row])

    return np.array(matrix)


def generatePoints(noise = 0.2, predictions = None):
    xValues = np.linspace(0, MAX_X, 200)
    yValues = np.cos(xValues)
    noiseVector = np.random.normal(scale=noise, size=yValues.shape)
    points = yValues + noiseVector
    return {"points": points,
            "xValues": xValues,
            "yValues" : yValues,
            }

def solveWeightsAnalytic(basisMatrix, targetPoints, ridge = False, lasso = False, lam = 20):
    ## func = phi(x) @ wT = preds
    ## find minimisation of 1/2 * (phi(x) @ wT - y) ** 2 (least squares)
    # since quad. solve for der. = 0
    # der = phi(x).T @ (phi(x) @ wT - y) = 0 # 2*0.5 disappears due to chain rule
    
    # so best wT = (phi(x).T @ phi(x))^-1 @ phi(x).T @ y

    ## Ridge 
    # find minimisation of 1/2 * (phi(x) @ wT - y) ** 2  + 1/2 * lam * sum(w**2)
    # der = phi(x).T @ (phi(x) @ wT - y) + lam * w = 0
    # phi(x).T @ phi(x) @ wT + lam *w = phi(x).T @ y
    # wT = (phi(x).T @ phi(x) + lam @ I)^-1 @ phi(x).T @ y
    if ridge:
        return np.linalg.inv(basisMatrix.T @ basisMatrix + lam * np.eye(basisMatrix.shape[1])) @ basisMatrix.T @ targetPoints
    if lasso:
        return "Error" # lasso not differ. at w=0
    return np.linalg.inv(basisMatrix.T @ basisMatrix) @ basisMatrix.T @ targetPoints

def plotPoints(xValues, yValues, points, predictions, currentDegree = -1, ax = None):
    rms = np.sqrt(np.sum((yValues - predictions)**2)/len(predictions))
    ax.plot(xValues, yValues, label = "True Underlying Function", color = "Orange")
    ax.scatter(xValues, points, label = "Noisy Points", color = "Red")
    ax.plot(xValues, predictions, label = "Predicted Function", color = "Green")
    if USE_GAUSSIAN_RBFS:
        basisFunc = "Gaussian RBF"
        bfOrDegree = "Basis Functions"
    else:
        basisFunc = "Polynomial RBF"
        bfOrDegree = "Degree"
    ax.set_title(f"{basisFunc} LR. RMS {round(rms, 2)} | {bfOrDegree} {currentDegree}")
    ax.legend()


if ANALYTIC_REGRESSION:
    if USE_GAUSSIAN_RBFS:
        start = 1
    else:
        start = 0
    ranges = range(start, DEGREES + 1, int(0.2 * DEGREES))
    rows = 2
    cols = int(np.ceil(len(ranges)/2))
    fig, axArray = plt.subplots(rows, cols, figsize=(12, 10))


    for deg in ranges:
        dictionaryOfDetails = generatePoints()
        points = dictionaryOfDetails["points"]
        xValues = dictionaryOfDetails["xValues"]
        targetFunction = dictionaryOfDetails["yValues"]

        basisM = generateBasisMatrix(xValues, degree=deg, gaussianRBF=USE_GAUSSIAN_RBFS)
        bestWeights = solveWeightsAnalytic(basisM, points, ridge=True)

        predictions = np.array(bestWeights @ basisM.T)   # hope this is the right way around

        plotPoints(xValues=xValues, yValues=targetFunction, points=points, predictions=predictions.flatten(), currentDegree=deg, ax=axArray.ravel()[list(ranges).index(deg)])

    plt.tight_layout()  
    plt.show()          

def gradientDescent(weights, basisMatrix, targetPoints, learningRate = LR):
    # if grad. positive, minima is "left", so push weight negative
    # vice versa
    # der = phi(x).T @ (phi(x) @ wT - y) 
    # ensure transposition is correct (numpy will broadcast you into oblivion if not)
    #
    part = (basisMatrix @ weights.reshape(-1,1) - targetPoints.reshape(-1, 1)) 
    grad = basisMatrix.T @ part
    return weights.reshape(-1,1) - learningRate * grad.reshape(-1,1)

def adaptiveMomentumEstimation(weights, basisMatrix, targetPoints, m, v, t, beta1=BETA_1, beta2=BETA_2, learningRate = LR, epsilon = 1e-8):
    part = (basisMatrix @ weights.reshape(-1,1) - targetPoints.reshape(-1, 1)) 
    grad = (basisMatrix.T @ part).reshape(-1, 1)

    m = beta1 * m.reshape(-1, 1) + (1-beta1) * grad # ewma of gradients, captures direction
    # first moment - m is the mean
    v = beta2 * v.reshape(-1, 1) + (1-beta2) * (grad) ** 2 # ewma of squared gradience, captures magnitude of gradients since we do ema of this
    # second moment - since Var = e(x**2) - e(x)**2 and we're doing squared x (gradients), and ema acts like an average
    # but e(x)**2 is not used here, so this becomes ROUGHLY e(x**2), so ROUGHLY the second moment.


    # bias correction - pushes m and v away from 0, but as time goes on mNorm  gets closer to m
    mNorm = m / (1 - beta1 ** t)
    vNorm = v / (1 - beta2 ** t)

    weights = weights.reshape(-1, 1) - (learningRate / (np.sqrt(vNorm) + epsilon)) * mNorm
    # denominator means that if gradients are large, we take smaller steps,aand if small, we take bigger stesps

    return weights.reshape(-1, 1), m, v


initialWeights = np.zeros(DEGREES + 1)

ITERS = 5000

generatedDetails = generatePoints()
points = generatedDetails["points"]
xValues = generatedDetails["xValues"]
targetValues = generatedDetails["yValues"]
basisM = generateBasisMatrix(xValues, degree=DEGREES, gaussianRBF=USE_GAUSSIAN_RBFS)


for iteration in range(1, ITERS+1):
    if iteration % 50 == 1:
        plt.plot(xValues, basisM @ initialWeights.reshape(-1, 1))
        plt.scatter(xValues, points, color = "Brown")
    # stuck, needs some sort of adaptive momentum such as Adam/Adagrad at least, or use gaussian RBFs
    initialWeights = gradientDescent(initialWeights, basisMatrix = basisM, targetPoints=points)

plt.plot(xValues, targetFunction, label = "Target Function")
plt.title("Standard Gradient Descent")
plt.legend()
plt.show()
plt.close()


initialWeights = np.zeros(DEGREES + 1)
m = np.zeros(DEGREES + 1)
v = np.zeros(DEGREES + 1)

for iteration in range(1, ITERS+1):
    if iteration % 50 == 1:
        plt.plot(xValues, basisM @ initialWeights.reshape(-1, 1))
        plt.scatter(xValues, points, color = "Brown")
    # stuck, needs some sort of adaptive momentum such as Adam/Adagrad at least, or use gaussian RBFs
    initialWeights, m, v = adaptiveMomentumEstimation(initialWeights, basisMatrix = basisM, targetPoints=points, m=m, v=v, t = iteration)

plt.plot(xValues, targetFunction, label = "Target Function")
plt.title("Adam Gradient Descent")
plt.legend()
plt.show()
plt.close()
