import numpy as np

def load_dataset(filename,delim='\t'):
    fr = open(filename)
    string_arr = [line.strip().split(delim) for line in fr.readlines()]
    dat_arr = [map(float,line) for line in string_arr]

    return np.mat(dat_arr)

def pca(data_mat,topNfeat=999999):
    # print np.shape(data_mat)
    mean_vals = np.mean(data_mat,axis=0)
    # print np.shape(mean_vals)
    mean_removed = data_mat - mean_vals

    cov_mat = np.cov(mean_removed,rowvar=0)

    eig_vals,eig_vects = np.linalg.eig(np.mat(cov_mat))
    eig_val_ind = np.argsort(eig_vals)
    eig_val_ind = eig_val_ind[:-(topNfeat):-1]
    red_eig_vects = eig_vects[:,eig_val_ind]

    low_data_mat = mean_removed * red_eig_vects
    recon_mat = (low_data_mat * red_eig_vects.T) +mean_vals

    return low_data_mat,recon_mat

data_mat = load_dataset(r'/home/wenjusun/bigdata/data/machinelearninginaction/Ch13/testSet.txt')
lowDMat,reconMat = pca(data_mat,1)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data_mat[:,0].flatten().A[0],data_mat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')

plt.show()
