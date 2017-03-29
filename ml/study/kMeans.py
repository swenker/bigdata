__author__ = 'wenjusun'

from numpy import *


def load_dataset(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr:
            current_line = line.strip().split("\t")
            flt_line = map(float, current_line)
            dataMat.append(flt_line)

    return dataMat


def dist_eclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def rand_cent(dataset, k):
    n = shape(dataset)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataset[:, j])
        rangeJ = float(max(dataset[:, j]) - minJ)
        # print random.rand(k, 1)
        # print minJ+rangeJ*random.rand(k,1)
        # print rangeJ
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)

        # print j,centroids
    return centroids


def kMeans(dataset, k, dist_meas=dist_eclud, create_cent=rand_cent):
    m = shape(dataset)[0]
    cluster_assment = mat(zeros((m, 2)))
    centroids = create_cent(dataset, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = dist_meas(centroids[j, :], dataset[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if cluster_assment[i, 0] != minIndex:
                cluster_changed = True
            cluster_assment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            ptsInClust = dataset[nonzero(cluster_assment[:, 0].A == cent)[0]]
            # print ptsInClust
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, cluster_assment


def biKmeans(dataset, k, dist_meas=dist_eclud):
    m = shape(dataset)[0]
    cluster_assment = mat(zeros((m, 2)))
    centroid0 = mean(dataset, axis=0).tolist()[0]
    cent_list = [centroid0]
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(mat(centroid0), dataset[j, :]) ** 2

    while (len(cent_list) < k):
        lowestSSE = inf
        for i in range(len(cent_list)):
            ptsInCurrCluster = dataset[nonzero(cluster_assment[:, 0].A == i)[0], :]
            centroidMat, split_clust_ass = kMeans(ptsInCurrCluster, 2, dist_meas)
            sse_split = sum(split_clust_ass[:, 1])
            sse_not_split = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])
            # print "sseSplit,and notSplit:", sse_split, sse_not_split
            if (sse_split + sse_not_split) < lowestSSE:
                best_cent_to_split = i
                best_new_cents = centroidMat
                best_clust_ass = split_clust_ass.copy()
                lowestSSE = sse_split + sse_not_split

        best_clust_ass[nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_clust_ass[nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent_to_split
        # print "the bestCentToSplit is :", best_cent_to_split
        # print "the len of bestClustAss is :", len(best_clust_ass)

        cent_list[best_cent_to_split] = best_new_cents[0, :]
        cent_list.append(best_new_cents[1, :])
        cluster_assment[nonzero(cluster_assment[:, 0].A == best_cent_to_split)[0], :] = best_clust_ass

    print type(cent_list)
    print cent_list
    return cent_list, cluster_assment


def test_it():
    dataset_file = r"C:\ZZZZZ\0-sunwj\bigdata\data\machinelearninginaction\Ch10\testSet2.txt"
    dataset_file = r"C:\ZZZZZ\0-sunwj\bigdata\data\machinelearninginaction\Ch10\testSet.txt"
    data_matrix = mat(load_dataset(dataset_file))
    # print dist_eclud(data_matrix[0],data_matrix[1])
    # rand_cent(data_matrix, 2)

    # myCentroids, clustAssing = kMeans(data_matrix, 4)
    myCentroids, clustAssing = biKmeans(data_matrix, 3)
    print myCentroids, clustAssing


if __name__ == "__main__":
    test_it()