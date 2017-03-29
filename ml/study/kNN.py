__author__ = 'wenjusun'

from numpy import *
import operator
from tools import *

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']

    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSet = autoNorm(dataSet)
    # print inX
    # inX = autoNorm(inX)
    # print inX,dataSet
    dataSetSize = dataSet.shape[0]
    inputArray = tile(inX,(dataSetSize,1))
    try:
        inputArray = autoNorm(inputArray)
    except RuntimeWarning:
        pass

    diffMat = inputArray - dataSet
    sqDiffMat = diffMat ** 2

    # print dataSet, tile(inX,(dataSetSize,1)),diffMat,sqDiffMat
    # print  tile(inX,(dataSetSize,1))

    sqDistances = sqDiffMat.sum(axis = 1)
    # sqDistances = sqDiffMat.sum()# sum all
    distances = sqDistances ** 0.5

    # Get the indices ,ascendancy
    sortedDistIndices = distances.argsort()

    # print distances,sqDistances,sortedDistIndices
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        # print voteIlabel,classCount[voteIlabel]

    print classCount
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse=True)
    print sortedClassCount
    return sortedClassCount[0][0]

# newValue = (oldValue-min)/(max-min)  thus the new value will be in [0,1]
def autoNorm(dataSet):

    minVals = dataSet.min(0) # min in every column to a new array :dataSet.shape(1)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    print dataSet, minVals,maxVals,ranges
    # normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))

    return normDataSet

def showDiagram():
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    groups = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    ax.scatter(groups[:1])
    ax.scatter(groups[:1],groups[:1])
    plt.show()


def imgbits2vector(imgbits):
    targetVector = zeros((1,1024))
    i=0
    for ib in imgbits:
        targetVector[0,i]=int(ib)
        if i==1024:
            break

    return targetVector

def img2vector(imgfile):
    "Convert 32*32 img to 1*1024 vector"
    targetVector = zeros((1,1024))
    with open(imgfile) as fr:
        i=0
        for line in fr:
            i+=1
            if i>32:
                break

            for j in range(32):
                print line[j]
                #targetVector[0,32*i +j] = hex(line[j])
    return targetVector

def testit():
    group,labels = createDataSet()
    print classify0([0,0],group,labels,2)
    # print classify0([0.8,0.9],group,labels,3)



# testit()
# autoNorm(createDataSet()[0])

# print img2vector(r'C:\ZZZZZ\tmp\2-1.bmp')
print imgbits2vector(get_bitmap(r'C:\ZZZZZ\0-sunwj\bigdata\data\bmp\2-1.bmp'))