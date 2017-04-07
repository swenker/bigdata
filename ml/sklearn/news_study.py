from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np

categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train= fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)

def show_some_data_attributes():
    # print twenty_train
    print twenty_train.target_names #holds the category name
    print len(twenty_train.data)
    print len(twenty_train.filenames)

    print "\n".join(twenty_train.data[0].split("\n")[:3])

    print twenty_train.target_names[twenty_train.target[0]]

    # the category integer id of each sample
    #twenty_train.target

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(twenty_train.data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print X_train_counts.shape

#feature indices
print count_vect.vocabulary_.get(u'algorithm')

print len(count_vect.vocabulary_)


def demo_classifier():
    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
    docs_new = ['God is love','OpenGL on the GPU is fast']
    docs_new = ['Then want to go to bed','There is parade']
    X_new_counts= count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

    print predicted

    for doc,category in zip(docs_new,predicted):
        print "%r => %s" %(doc,twenty_train.target_names[category])



def bayes_classifier():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    print np.mean(predicted == twenty_test.target)

def SVM_classifier():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,n_iter=5,random_state=42))])
    text_clf.fit(twenty_train.data, twenty_train.target)

    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    print np.mean(predicted == twenty_test.target)

    print metrics.classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names)
    print metrics.confusion_matrix(twenty_test.target,predicted)

SVM_classifier()

