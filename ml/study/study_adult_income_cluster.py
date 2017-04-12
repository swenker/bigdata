import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_kmeans():
    from adult_data_utils import DataDict
    from adult_income_classification import DataParser

    data_dict = DataDict()
    data_parser = DataParser()
    filepath='/home/wenjusun/bigdata/data/adult-income/adult.data'
    limit = 1000
    # X,real_labels=data_dict.get_training_set()
    record_list = data_parser.parse_file_fetch_records(filepath, limit)
    X = np.array(data_parser.records_to_vector(record_list, enable_label=True))

    kmeans_model = KMeans(n_clusters=4).fit(X)
    labels = kmeans_model.labels_
    print kmeans_model.cluster_centers_
    # print labels[:100]
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


cluster_kmeans()
