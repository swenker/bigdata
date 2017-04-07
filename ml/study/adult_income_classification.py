__author__ = 'wenjusun'

"https://artax.karlin.mff.cuni.cz/r-help/library/arules/html/Adult.html"

import operator
from math import log

from numpy import *
from adult_data_utils import DataDict

data_dict = DataDict()
# LABEL_IND=

class DataParser:
    AGE_MAX = 90
    AGE_MIN = 17
    AGE_MEAN = 38.58
    AGE_SD = 13.64   #awk '{sum+=$1;sumsq+=($1)^2} END { print sqrt((sumsq-sum^2/NR)/NR) }'

    CAPITAL_GAIN_MAX = 9999
    CAPITAL_GAIN_MEAN = 1077.65
    CAPITAL_GAIN_SD = 7463.29

    CAPITAL_LOSS_MAX = 4356
    CAPITAL_LOSS_MEAN = 87.30
    CAPITAL_LOSS_SD = 410.49
    CAPITAL_MIN = 0

    HOURS_PER_WEEK_MAX = 99
    HOURS_PER_WEEK_MIN = 1
    HOURS_PER_WEEK_MEAN = 40.44
    HOURS_PER_WEEK_SD = 17.29

    def parse_file_fetch_records(self, filename, limit=None):
        record_list = []
        with open(filename, mode='r') as data_file:
            i = 0
            # with open(filename, mode='r', encoding='UTF-8') as data_file:
            for line in data_file:
                # print i
                i += 1
                line = line.strip()
                if line:
                    line = line.replace('?', '')
                    fields = line.split(',')
                    data_record = DataRecord(fields)
                    record_list.append(data_record)
                if limit and i >= limit:
                    break
        return record_list

    def parse_file(self, filename):
        record_list = []
        with open(filename, mode='r') as data_file:
            i = 1
            # with open(filename, mode='r', encoding='UTF-8') as data_file:
            for line in data_file:
                # print i
                # i+=1
                line = line.strip()
                if line:
                    line = line.replace('?', '')
                    fields = line.split(',')
                    data_record = DataRecord(fields)
                    record_list.append(data_record)
        return record_list

    def __features_to_vector(self, data_record, ENABLE_LABEL=False):
        "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K"
        feature_vector = []
        age = DataDict.rescale_continuous_to_1(data_record.age, self.AGE_MAX, self.AGE_MIN,self.AGE_MEAN,self.AGE_SD)
        feature_vector.append(age)
        # fntl
        feature_vector.extend(data_dict.get_workclass_vector(data_record.workclass))
        feature_vector.extend(data_dict.get_education_vector(data_record.education))
        # TODO edu num
        feature_vector.extend(data_dict.get_marital_status_vector(data_record.marital_status))
        feature_vector.extend(data_dict.get_occupation_vector(data_record.occupation))
        feature_vector.extend(data_dict.get_relationship_vector(data_record.relationship))
        feature_vector.extend(data_dict.get_race_vector(data_record.race))
        feature_vector.extend(data_dict.get_sex_vector(data_record.sex))

        capital_gain = DataDict.rescale_continuous_to_1(data_record.capital_gain, self.CAPITAL_GAIN_MAX,
                                                        self.CAPITAL_MIN,self.CAPITAL_GAIN_MEAN,self.CAPITAL_GAIN_SD)
        feature_vector.append(capital_gain)

        capital_loss = DataDict.rescale_continuous_to_1(data_record.capital_loss, self.CAPITAL_LOSS_MAX,
                                                        self.CAPITAL_MIN,self.CAPITAL_LOSS_MEAN,self.CAPITAL_LOSS_SD)
        feature_vector.append(capital_loss)

        hours_pe_week = DataDict.rescale_continuous_to_1(data_record.hours_per_week, self.HOURS_PER_WEEK_MAX,
                                                         self.HOURS_PER_WEEK_MIN,self.HOURS_PER_WEEK_MEAN,self.HOURS_PER_WEEK_SD)
        feature_vector.append(hours_pe_week)

        feature_vector.extend(data_dict.get_native_country(data_record.native_country))

        if ENABLE_LABEL:
            feature_vector.append(data_dict.get_income_large_50k(data_record.label.rstrip('.')))

        return feature_vector

    def records_to_vector(self, record_list, enable_label=False):
        records_vector = []
        for record in record_list:
            records_vector.append(self.__features_to_vector(record, enable_label))
        return records_vector


class DataRecord:
    age = 0
    workclass = ''
    fnlwgt = 0
    education = ''
    education_num = 0
    marital_status = ''
    occupation = ''
    relationship = ''
    race = ''
    sex = ''
    capital_gain = 0
    capital_loss = 0
    hours_per_week = 0
    native_country = ''
    label = ''

    def __init__(self, field_array):
        "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K"
        if field_array:
            self.age = int(field_array[0].strip())
            self.workclass = field_array[1].strip()
            self.fnlwgt = int(field_array[2].strip())
            self.education = field_array[3].strip()
            self.education_num = int(field_array[4].strip())
            self.marital_status = field_array[5].strip()
            self.occupation = field_array[6].strip()
            self.relationship = field_array[7].strip()
            self.race = field_array[8].strip()
            self.sex = field_array[9].strip()
            self.capital_gain = int(field_array[10].strip())
            self.capital_loss = int(field_array[11].strip())
            self.hours_per_week = int(field_array[12].strip())
            self.native_country = field_array[13].strip()
            self.label = field_array[14].strip()


filepath = r'C:\ZZZZZ\0-sunwj\bigdata\data\adult-income\adult.data'
data_parser = DataParser()


def test_it():
    data_dict = DataDict()
    # print data_dict.get_workclass_vector('State-gov')
    filepath = r'C:\ZZZZZ\0-sunwj\bigdata\data\adult-income\adult-2.data'
    record_list = data_parser.parse_file(filepath)
    f_vector_list = data_parser.records_to_vector(record_list)
    # print len(f_vector_list[0])
    for v in f_vector_list:
        print "%d," % v[-1],


# test_it()

# import decision_tree

class Classifier():
    def kNNClassifier(self, inX, knownDataSet, knownLabels, k):
        """
        :param inX: only one line
        :param knownDataSet:
        :param knownLabels:
        :param k:
        :return:
        """
        dataSetSize = knownDataSet.shape[0]
        # print dataSetSize
        inputArray = tile(inX, (dataSetSize, 1))

        diffMat = inputArray - knownDataSet
        sqDiffMat = diffMat ** 2

        sqDistances = sqDiffMat.sum(axis=1)
        # sqDistances = sqDiffMat.sum()# sum all
        distances = sqDistances ** 0.5

        # Get the indices ,ascendancy
        sortedDistIndices = distances.argsort()

        # print distances,sqDistances,sortedDistIndices
        classCount = {}
        for i in range(k):
            voteIlabel = knownLabels[sortedDistIndices[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
            # print voteIlabel,classCount[voteIlabel]

        # print classCount
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        # print sortedClassCount
        return sortedClassCount[0][0]

    def createTree(self,data_set,labels):
        return decision_tree.createTree(data_set,labels)

    def bayes_train(self, train_set, train_labels):
        feature_num = len(train_set[0])
        train_set_len = len(train_set)
        # p_income_larger = 1.0 * 7841 / 24720
        p_income_larger = 1.0 *  sum(train_labels)/ train_set_len
        p0Num = ones(feature_num)
        p1Num = ones(feature_num)

        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(train_set_len):
            if train_labels[i] == 1:
                p1Num += train_set[i]
                p1Denom += sum(train_set[i])
            else:
                p0Num += train_set[i]
                p0Denom += sum(train_set[i])

        p1Vect = log(absolute(p1Num) / p1Denom)
        p0Vect = log(absolute(p0Num) / p0Denom)

        return p0Vect, p1Vect, p_income_larger


    def bayes_classifier(self, vec2Classify, p0Vec, p1Vec, pClass1):
        p1 = sum(vec2Classify * p1Vec) + log(pClass1)
        p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
        if (p1 > p0):
            return 1
        else:
            return 0

filepath = r'C:\ZZZZZ\0-sunwj\bigdata\data\adult-income\adult.data'

classifier = Classifier()


def test_kNN():
    train_set, train_labels = data_dict.get_training_set()
    labels = data_dict.get_all_labels(label_file = r'C:\ZZZZZ\0-sunwj\bigdata\data\adult-income\test-labels.txt')
    record_list = data_parser.parse_file(filepath)
    f_vector_list = data_parser.records_to_vector(record_list)

    for K in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        target_labels = []
        for v in f_vector_list:
            inX = array(v)
            cal_label = classifier.kNNClassifier(inX, array(train_set), train_labels, K)
            target_labels.append(cal_label)

        assert len(target_labels) == len(labels)

        error_count = 0
        for i in range(0, len(target_labels)):
            if target_labels[i] != labels[i]:
                error_count += 1

        print "K=%d,error:%d,%f%%" % (K, error_count, 1.0 * error_count / len(labels))

def get_train_set(limit):
    filepath = r'C:\ZZZZZ\0-sunwj\bigdata\data\adult-income\adult.test'
    record_list = data_parser.parse_file_fetch_records(filepath, limit)
    features_vector_withlabel = array(data_parser.records_to_vector(record_list, enable_label=True))

    candidate_labels = features_vector_withlabel[:,-1]
    candidate_set = delete(features_vector_withlabel, -1, axis=1)

    return candidate_set,candidate_labels

def test_decision_tree():
    feature_labels = data_dict.get_feature_labels()
    record_list = data_parser.parse_file_fetch_records(filepath, 100)
    features_vector_withlabel = data_parser.records_to_vector(record_list, enable_label=True)
    labels_copy = feature_labels[:]
    generated_decision_tree=classifier.createTree(features_vector_withlabel,feature_labels)
    print generated_decision_tree

    decision_tree.createTreePlot(generated_decision_tree)
    dataset,class_label = get_train_set(10)
    print class_label
    for candidate_record in dataset:
        ""
        # class_value=classifier.decision_tree_classifier(generated_decision_tree,labels_copy,candidate_record)
        # class_value=decision_tree.classify(generated_decision_tree,labels_copy,candidate_record)

        # print class_value

def test_bayes():
    # record_list = data_parser.parse_file_fetch_records(filepath, 5)
    record_list = data_parser.parse_file_fetch_records(filepath)
    features_vector_withlabel = array(data_parser.records_to_vector(record_list, enable_label=True))
    # print features_vector_withlabel

    candidate_labels = features_vector_withlabel[:, -1]
    candidate_set = delete(features_vector_withlabel, -1, axis=1)

    # p0Vec, p1Vec, pClass1 = classifier.bayes_train(train_set, labels)

    for K in [20,50,100,200,500,1000,2000,5000,10000,15000]:
        train_set, train_labels = get_train_set(K)
        p0Vec, p1Vec, pClass1 = classifier.bayes_train(array(train_set), array(train_labels))
        target_labels = []
        for vr in candidate_set:
            target_labels.append(classifier.bayes_classifier(array(vr), p0Vec, p1Vec, pClass1))

        assert len(target_labels) == len(candidate_labels)
        # print target_labels
        # print labels
        error_count = 0
        for i in range(0, len(target_labels)):
            if target_labels[i] != candidate_labels[i]:
                error_count += 1

        print "K=%d,error:%d,%f%%" % (K, error_count, 1.0 * error_count / len(candidate_labels))

def test_usage():
    feature_labels = data_dict.get_feature_labels()
    print len(feature_labels)
    print feature_labels

def test_classifier():
    filepath = r'C:\ZZZZZ\0-sunwj\bigdata\data\adult-income\adult-00-test.txt'

    # test_kNN()
    # test_decision_tree()
    test_bayes()
    # test_usage()

if __name__=='__main__':
    test_classifier()

