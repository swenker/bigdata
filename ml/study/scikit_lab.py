__author__ = 'wenjusun'

from numpy import *

from sklearn import datasets
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn import metrics


def get_dataset():
    pass

def kNN():
    samples = array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    nbrs = neighbors.NearestNeighbors(n_neighbors=4).fit(samples)
    X=[[1,0]]
    distances, indices = nbrs.kneighbors(X)
    print distances,indices

def cluster_kmeans1():
    dataset = datasets.load_iris()
    X = dataset.data
    kmeans_model = KMeans(n_clusters=4,random_state=1).fit(X)

    labels = kmeans_model.labels_
    print X
    print kmeans_model.cluster_centers_
    print labels
    print metrics.silhouette_score(X,labels,metric="euclidean")

def cluster_kmeans_simple():
    from adult_data_utils import DataDict
    data_dict = DataDict()
    X,real_labels=data_dict.get_training_set()

    kmeans_model = KMeans(n_clusters=2).fit(X)
    labels = kmeans_model.labels_

    print labels
    print array(real_labels)

    array_len = len(labels)
    equal_count = 0
    for i in range(array_len):
        if labels[i] == real_labels[i]:
            equal_count+=1

    difference = array(real_labels)-labels
    error_count = array_len-equal_count
    print "eq:%d,err:%d" %(equal_count , error_count)
    print error_count*1./array_len

def cluster_kmeans():
    from adult_data_utils import DataDict
    from adult_income_classification import DataParser

    data_dict = DataDict()
    data_parser = DataParser()
    filepath='/home/wenjusun/bigdata/data/adult-income/adult.data'
    limit = 20
    # X,real_labels=data_dict.get_training_set()
    record_list = data_parser.parse_file_fetch_records(filepath, limit)
    X = array(data_parser.records_to_vector(record_list, enable_label=True))

    kmeans_model = KMeans(n_clusters=2).fit(X)
    labels = kmeans_model.labels_

    print labels
    print X[:,-1]
    # print array(real_labels)


def testit():
    # kNN()
    # cluster_kmeans()
    cluster_kmeans_simple()

if __name__ == "__main__":
    testit()