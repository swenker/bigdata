import numpy as np
import matplotlib.pyplot as plt

from adult_data_utils import DataDict
from adult_income_classification import DataParser
from time import time
start = time()

data_dict = DataDict()
data_parser = DataParser()

def get_training_data(limit):
    filepath = '/home/wenjusun/bigdata/data/adult-income/adult.data'
    record_list = data_parser.parse_file_fetch_records(filepath, limit)
    income_data = np.array(data_parser.records_to_vector(record_list, enable_label=True))
    X = income_data[:, :-1]
    original_labels = income_data[:, -1]
    return X,original_labels

def get_test_data(limit):
    filepath = '/home/wenjusun/bigdata/data/adult-income/adult.test'
    record_list = data_parser.parse_file_fetch_records(filepath, limit)
    income_data = np.array(data_parser.records_to_vector(record_list, enable_label=True))
    X = income_data[:, :-1]
    original_labels = income_data[:, -1]
    return X,original_labels

def classification_knn():
    limit =10000
    X,original_labels = get_training_data(limit)
    from sklearn.neighbors import KNeighborsClassifier

    knc = KNeighborsClassifier(n_neighbors=5)
    knc.fit(X,original_labels)

    print "time passed for training data:%d" %(time()-start)

    limit =5000 # error:n=3,e=0.183;n=5,e=0.173
    X,original_labels = get_test_data(limit)
    result = knc.predict(X)
    show_classification_result(original_labels,result,limit)

def classification_bayes():
    limit =10000
    X,original_labels = get_training_data(limit)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import MultinomialNB
    estimator = GaussianNB() #5000:0.617
    # estimator = BernoulliNB() #5000: 0.235
    # estimator  = MultinomialNB() #5000:0.179
    estimator.fit(X,original_labels)

    limit =5000 # error:
    X,original_labels = get_test_data(limit)
    result = estimator.predict(X)
    show_classification_result(original_labels,result,limit)

def classification_svm():
    limit =10000
    X,original_labels = get_training_data(limit)
    from sklearn import svm
    estimator  = svm.SVC() #5000:0.153
    estimator.fit(X,original_labels)
    print "time passed for training data:%d" %(time()-start)
    limit =5000 # error:
    X,original_labels = get_test_data(limit)
    result = estimator.predict(X)
    show_classification_result(original_labels,result,limit)


def classification_decision_tree():
    limit =10000
    X,original_labels = get_training_data(limit)
    from sklearn import tree
    estimator  = tree.DecisionTreeClassifier() #5000:0.187
    estimator.fit(X,original_labels)
    print "time passed for training data:%d" %(time()-start)
    limit =5000 # error:
    X,original_labels = get_test_data(limit)
    result = estimator.predict(X)
    show_classification_result(original_labels,result,limit)

def show_classification_result(original_labels,predicted_labels,n_records):
    error_diff = original_labels - predicted_labels
    print len(error_diff[np.nonzero(error_diff)])*1.0/n_records


def cluster_kmeans():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    # import sklearn.decomposition.pca

    limit = 10000
    # X,real_labels=data_dict.get_training_set()
    filepath = '/home/wenjusun/bigdata/data/adult-income/adult.data'
    record_list = data_parser.parse_file_fetch_records(filepath, limit)
    X = np.array(data_parser.records_to_vector(record_list, enable_label=False))

    pca_estimator = PCA(n_components=1)

    X=pca_estimator.fit_transform(X)

    kmeans_model = KMeans(n_clusters=4).fit(X)
    labels = kmeans_model.labels_
    # print kmeans_model.cluster_centers_
    # print labels[:100]
    print len(X),len(labels)
    print labels[:40]
    # print array(real_labels)

    # count=0
    # for xLabel,eLabel in zip(X[-1],labels):
    #     if xLabel==eLabel:
    #         count +=1
    #
    # print "count=%d,ratio:%f" %(count,1.0*count/len(labels))
    # print np.sum(labels)
    plt.figure(1)
    plt.scatter(X,labels)
    plt.show()


# cluster_kmeans()
classification_knn()
# classification_bayes()
# classification_svm()
# classification_decision_tree()
end = time()
print " time passed:%d" %(end - start)
