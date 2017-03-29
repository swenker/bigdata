__author__ = 'wenjusun'

from numpy import *
import feedparser

from document_filter import *

def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
        # print vocabSet
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        # else:
        #     print "The word:%s is not in my vocabulary " % word
    return returnVec


def bagOfWords2VecMn(vocabList, inputSet):
    "input set to vocabulary length vector with count"
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        # else:
        #     print "The word:%s is not in my vocabulary " % word
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    p0Num = ones(numWords)
    p1Num = ones(numWords)

    p0Denom = 2.0
    p1Denom = 2.0

    # print dtype(p0Num)
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom

    print p1Num/p1Denom

    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    # print p1Vect
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if (p1 > p0):
        return 1
    else:
        return 0


def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postingDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb)

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb)


def testit_old():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)

    # print myVocabList
    # print setOfWords2Vec(myVocabList,listPosts[0])
    trainMat = []
    for postingDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    # print trainMat
    # print listClasses
    p0v, p1v, pAb = trainNB0(trainMat, listClasses)
    print p0v, p1v
    # print pAb

def textParse( bigString):
    import re

    listOfTokens = re.split(r'\W*', bigString)
    # return [token.lower() for token in listOfTokens if len(token) > 2]
    return [token.lower() for token in listOfTokens if not stopWordsFilter.is_stopword(token)]

class SpamFilter:

    def spamTest(self):
        docList = []
        classList = []
        fullText = []

        for i in range(1, 26):
            spamString = ''
            hamString = ''
            wordList = textParse(spamString)
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            wordList = textParse(hamString)
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = createVocabList(docList)
        trainingSet = range(50)
        testSet = []
        for i in range(10):
            randIndex = int(random.uniform(0, len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del (trainingSet[randIndex])
        trainMat = []
        trainClasses = []
        for docIndex in trainingSet:
            trainMat.append(bagOfWords2VecMn(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

        errorCount = 0
        for docIndex in testSet:
            wordVector = setOfWords2Vec(vocabList, docList[docIndex])
            if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1
        print 'The error rage is :', float(errorCount) / len(testSet)


class RssFeedParser:
    _one_underscore=1
    __two_underscore=2

    def _one(self):
        print "I am one"
    def __two(self):
        print "I am two"

    def calcMostFreq(self, vocabList, fullText):
        import operator

        freqDict = {}
        for token in vocabList:
            freqDict[token] = fullText.count(token)
        sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
        # print sortedFreq[:30]
        return sortedFreq[:30]

    def localWords(self,feed1,feed0):
        docList=[]
        classList=[]
        fullText = []

        minLen = min(len(feed1['entries']),len(feed0['entries']))

        for i in range(minLen):
            wordList = textParse(feed1['entries'][i]['summary'])
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)

            wordList = textParse(feed0['entries'][i]['summary'])
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = createVocabList(docList)
        print vocabList

        "remove top words from vocab"
        topWords = self.calcMostFreq(vocabList,fullText)
        for pairW in topWords:
            if pairW[0] in vocabList:
                vocabList.remove(pairW[0])

        trainingSet = range(2* minLen)
        testSetIndex=[]
        for i in range(20):
            randIndex = int(random.uniform(0,len(trainingSet)))
            testSetIndex.append(trainingSet[randIndex])
            del[trainingSet[randIndex]]
        # print testSetIndex
        trainMat = []
        trainClasses=[]
        for docIndex in trainingSet:
            docVec = bagOfWords2VecMn(vocabList,docList[docIndex])
            # print docVec
            trainMat.append(docVec)
            trainClasses.append(classList[docIndex])
        p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))

        errorCount = 0
        for docIndex in testSetIndex:
            wordVector = bagOfWords2VecMn(vocabList,docList[docIndex])
            if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
                errorCount +=1

        print 'the error rate is :' ,float(errorCount)/len(testSetIndex)
        return vocabList,p0V,p1V

    def getTopWords(self,ny,sf):
        import operator
        vocabList,p0V,p1V = self.localWords(ny,sf)
        topNY=[]
        topSF=[]
        for i in range(len(p0V)):
            if p0V[i]>-6.0:
                topSF.append((vocabList[i],p0V[i]))
            if p1V[i]>-6.0:
                topSF.append((vocabList[i],p1V[i]))

        sortedSF = sorted(topSF,key=operator.itemgetter(1),reverse=True)
        print 'SF**'*6
        for item in sortedSF:
            print item[0]

        sortedNY= sorted(topNY,key=operator.itemgetter(1),reverse=True)
        print 'NY**'*6
        for item in sortedNY:
            print item[0]

def test_classify_from_feed():
     rssParser = RssFeedParser()
     rssUrl_ny = 'http://newyork.craigslist.org/stp/index.rss'
     rssUrl_sf = 'http://sfbay.craigslist.org/stp/index.rss'
     rss_ny = feedparser.parse(rssUrl_ny)
     rss_sf = feedparser.parse(rssUrl_sf)
     vocabList,pSF,pNY = rssParser.localWords(rss_ny,rss_sf)
     # vocabList,pSF,pNY = rssParser.localWords(rss_ny,rss_sf)

     # rssParser.getTopWords(rss_ny,rss_sf)

testingNB()
# test_classify_from_feed()
