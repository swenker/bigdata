__author__ = 'wenjusun'
from numpy import *


def load_dataset():
    dataMat = []
    labelMat = []
    filepath = r'C:\ZZZZZ\0-sunwj\bigdata\data\machinelearninginaction\Ch05\testSet.txt'
    with open(filepath) as fr:
        for line in fr:
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))

    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + ma.exp(-inX))


def gradientAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001

    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights - alpha * dataMatrix.transpose() * error

    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt

    dataMat, labelMat = load_dataset()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        value1 = dataArr[i, 1]
        value2 = dataArr[i, 2]
        if int(labelMat[i]) == 1:
            xcord1.append(value1)
            ycord1.append(value2)
        else:
            xcord2.append(value1)
            ycord2.append(value2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel(('X1'))
    plt.ylabel(('X2'))
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def testit():
    dataArr, labelMat = load_dataset()
    # result = gradientAscent(dataArr, labelMat)
    # plotBestFit(result.getA())
    # result = stocGradAscent0(array(dataArr), labelMat)
    # result = stocGradAscent1(array(dataArr), labelMat)
    result = stocGradAscent1(array(dataArr), labelMat, 500)
    plotBestFit(result)


if __name__ == "__main__":
    testit()
