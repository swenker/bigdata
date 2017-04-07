import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X = X[ Y !=0,:2]
Y = Y[ Y!=0]

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
Y = Y[order].astype(np.float)

index_fence = n_sample-n_sample/10
X_train = X[:index_fence ]
Y_train = Y[:index_fence]
X_test = X[index_fence:]
Y_test = Y[index_fence:]

for fig_num,kernel in enumerate(('linear','rbf','poly')):
    clf = svm.SVC(kernel=kernel,gamma=10)
    clf.fit(X_train,Y_train)

    # figure title
    plt.figure(fig_num)
    plt.clf()

    plt.scatter(X[:,0],X[:,1],c=Y,zorder=10,cmap=plt.cm.Paired)

    plt.scatter(X_test[:,0],X_test[:,1],s=80,facecolors='none',zorder=10)
    plt.axis('tight')
    x_min = X[:,0].min()
    x_max = X[:,0].max()
    y_min = X[:,1].min()
    y_max = X[:,1].max()

    XX,YY = np.mgrid[x_min:x_max:200j,y_min:y_max:200j]
    Z=clf.decision_function(np.c_[XX.ravel(),YY.ravel()])
    Z=Z.reshape(XX.shape)

    plt.pcolormesh(XX,YY,Z >0,cmap=plt.cm.Paired)
    plt.contour(XX,YY,Z,colors=['k','k','k'],linestyles=['--','-','--'],levels=[-.5,0,.5])
    plt.title(kernel)

plt.show()
