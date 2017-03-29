__author__ = 'wenjusun'

from numpy import *
import math


def load_simpledata():
    datMat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])

    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return datMat, classLabels


def stump_classfy(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray


def build_stump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf

    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stump_classfy(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print "split:dim %d,thresh %.2f,thresh inequal:%s,the weighted error is %.3f" % (i, threshVal, inequal, weightedError)

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClasEst


def ada_boost_trainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))

    for i in range(numIt):
        bestStump, error, classEst = build_stump(dataArr, classLabels, D)
        print "D:", D.T
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst:", classEst.T
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        # TypeError: only length-1 arrays can be converted to Python scalars if math.exp,use np.exp instead
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print "aggClassEst:", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error:", errorRate, "\n"
        if errorRate == 0.0:
            break
    # return weakClassArr
    return weakClassArr,aggClassEst

def ada_classify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stump_classfy(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst

        print aggClassEst
    return sign(aggClassEst)


dataMat, classLabels = load_simpledata()

def plot_roc(pred_strengths,class_labels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum =1.0
    numPosClas = sum(array(class_labels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(class_labels)-numPosClas)
    sortedIndicies = pred_strengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot()
    for index in sortedIndicies.tolist()[0]:
        if class_labels[index] ==1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    ax.axis([0,1,0,1])
    plt.show()
    print "The area under the curve is :", ySum * xStep

def load_horses_test_data(filepath):
    dataMat = []
    labelMat = []

    num_feature = 22
    with open(filepath) as fr:
        for line in fr:
            line_array = []
            fields_array = line.strip().split('\t')
            for i in range(num_feature-1):
                line_array.append(float(fields_array[i]))
            dataMat.append(line_array)
            labelMat.append(float(fields_array[-1]))
    return dataMat, labelMat

def test_build_stump():
    D = mat(ones((5, 1)) / 5)
    print dataMat, classLabels
    print build_stump(dataMat, classLabels, D)

def test_it():
    classifierArray = ada_boost_trainDS(dataMat, classLabels, 9)
    print classifierArray

def test_classifier():
    classifierArray = ada_boost_trainDS(dataMat, classLabels, 30)
    print ada_classify([0,0],classifierArray)

def test_horses():
    filepath = r'C:\ZZZZZ\0-sunwj\bigdata\data\machinelearninginaction\Ch05\horseColicTraining.txt'
    dataArr,labelArr = load_horses_test_data(filepath)
    classifierArray = ada_boost_trainDS(dataArr,labelArr,10)

    filepath = r'C:\ZZZZZ\0-sunwj\bigdata\data\machinelearninginaction\Ch05\horseColicTest.txt'
    test_dataArr,test_labelArr = load_horses_test_data(filepath)
    prediction10 = ada_classify(test_dataArr,classifierArray)
    err_arr = mat(ones((67,1)))
    print err_arr[prediction10!=mat(test_labelArr).T].sum()

def test_show_roc():
    filepath = r'C:\ZZZZZ\0-sunwj\bigdata\data\machinelearninginaction\Ch05\horseColicTraining.txt'
    dataArr,labelArr = load_horses_test_data(filepath)
    classifierArray,aggClassEst = ada_boost_trainDS(dataArr,labelArr,10)
    plot_roc(aggClassEst.T,labelArr)

if __name__ == "__main__":
    # test_it()
    # test_build_stump()
    # test_classifier()
    # test_horses()
    test_show_roc()


