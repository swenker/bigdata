from sklearn import datasets,neighbors,linear_model
import numpy as np

digits = datasets.load_digits()
X_digits = digits.data
Y_digits = digits.target

n_samples = len(X_digits)

print("%f,%f" %(.9*n_samples,n_samples - n_samples/9))

data_fence = n_samples - n_samples/9
X_train = X_digits[: data_fence]
Y_train = Y_digits[:n_samples - n_samples/9]
X_test = X_digits[n_samples - n_samples/9:]
Y_test = Y_digits[n_samples - n_samples/9:]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()

classifier = knn.fit(X_train,Y_train)
predict_result = knn.predict(X_test)
# print Y_test
# print np.mean(np.array(predict_result),np.array(Y_test))

right_ones=0
for x,y in zip(predict_result,Y_test):
    print x,y
    if y==x:
        right_ones +=1

print ("%d,%d,%f" %(right_ones,len(Y_test),right_ones*1.0/len(Y_test)))

#TODO how to tell a new item's target?
# print('KNN score: %f' %knn.fit(X_train,Y_train).score(X_test,Y_test))

# print('LogisticRegression score:%f' % logistic.fit(X_train,Y_train).score(X_test,Y_test))