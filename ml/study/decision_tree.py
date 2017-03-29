__author__ = 'wenjusun'

import operator
from math import log
import matplotlib.pyplot as plt


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    # count all the labels
    for featVec in dataSet:
        currentLabel = featVec[-1]

        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0

        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    # shannonEnt = -[the summary of pX*log pX]
    for key in labelCounts:
        probability = float(labelCounts[key]) / numEntries
        shannonEnt -= probability * log(probability, 2)

    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'],
               # [1, 1, 'maybe'],
    ]
    dataSet2 = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [0, 1, 'no'],
                [1, 0, 'no'],
                [1, 0, 'no'],
                # [1, 1, 'maybe'],
    ]

    labels = ['no surfacing', 'flippers']

    # print labels
    return dataSet, labels


# Get a new data set matching the value and the result does not contain the value column anymore.
# 1,1,'yes'  : 1,1
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # print numFeatures,baseEntropy
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


def majorityCnt(classList):
    """  Get the label name with most references """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

ab=0
def createTree(dataSet, labels):
    global ab
    ab = ab+1
    # print dataSet
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # print "Here 1:same class now."
        return classList[0]
    if len(dataSet[0]) == 1:
        # print "Here 2"
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    # print bestFeat
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}

    del (labels[bestFeat])
    festValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(festValues)
    # print "I am :%d,feat:%d,class_val_len:%d " %(ab,bestFeat,len(uniqueVals))

    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createTreePlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def createPlot_old():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot_old.ax1 = plt.subplot(111, frameon=False)
    plotNode('DecisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('LeafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
    ]

    return listOfTrees[i]

def createTreePlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createTreePlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createTreePlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    for key in secondDict.keys():
        if type(secondDict[key]) is dict:
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    print "=",inputTree.keys()
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) is dict :
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]

    return classLabel

def storeTree(inputTree,filename):
    "save object to file"
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    "Read object from file"
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def testit():
    myDat, labels = createDataSet()

    # print calcShannonEnt(myDat)
    # print splitDataSet(myDat,2,'no')
    # print splitDataSet(myDat,0,0)

    # print chooseBestFeatureToSplit(myDat)
    labels_copy = labels[:]
    myTree = createTree(myDat, labels_copy)
    # print labels
    print myTree
    # print labels[0]
    # myTree = retrieveTree(0)
    # print getNumLeafs(myTree)
    # print getTreeDepth(myTree)

    print classify(myTree,labels,[1,1])
    # print classify(myTree,labels,[1,0])


def test_plot():
    # myTree = retrieveTree(0)
    myDat, labels = createDataSet()
    labels_copy = labels[:]
    myTree = createTree(myDat, labels_copy)

    createTreePlot(myTree)

def test_serializable():
    """

    :rtype : object
    """
    myTree= retrieveTree(0)
    filename='mytree.txt'
    storeTree(myTree,filename)
    print grabTree(filename)
if __name__=="__main__":
    # test_serializable()
    testit()
    # test_plot()