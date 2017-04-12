__author__ = 'wenjusun'

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def plot_hyperplane(clf,min_x,max_x,linestyle,label):
    w = clf.coef_[0]
    a = -w[0]/w[1]

    xx = np.linspace(min_x-5,max_x +5)
    yy = a * xx - (clf.intercept_[0])/w[1]

    plt.plot(xx,yy,linestyle,label=label)

def plot_subfigure(X, Y, subplot_index, subplot_title, transform):
    if transform =='pca':
        X=PCA(n_components=2).fit_transform(X)

    elif transform =='cca':
        X = CCA(n_components=2).fit(X,Y).transform(X)
    else:
        raise  ValueError

    min_x = np.min(X[:,0])
    max_x = np.max(X[:,0])

    min_y = np.min(X[:,1])
    max_y = np.max(X[:,1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X,Y)

    plt.subplot(2, 2, subplot_index)
    plt.title(subplot_title)

    zero_class = np.where(Y[:,0])
    one_class = np.where(Y[:,1])

    # print zero_class
    # print one_class
    plt.scatter(X[:,0],X[:,1],s=40,c="gray")
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
               facecolors='none', linewidths=2, label='Class 1')

    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
               facecolors='none', linewidths=2, label='Class 2')

    #-------------------Hyperplane.....
    plot_hyperplane(classif.estimators_[0],min_x,max_x,'k--','Boundary \n for class 1')
    plot_hyperplane(classif.estimators_[1],min_x,max_x,'k--','Boundary \n for class 2')

    # plt.xticks()
    # locs, labels = plt.yticks()

    # value range of the axis
    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    print int(min_x - .5 * max_x),int(min_x + .5 * max_x)
    print int(min_y - .5 * max_y), int(min_y + .5 * max_y)

    if subplot_index == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")

plt.figure(figsize=(8, 6))

#Generate a random multilabel classification problem
X, Y = make_multilabel_classification(n_classes=2, n_labels=1, allow_unlabeled=True, random_state=1)

# print X,Y

plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

X, Y = make_multilabel_classification(n_classes=2, n_labels=1, allow_unlabeled=False, random_state=1)

plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()
