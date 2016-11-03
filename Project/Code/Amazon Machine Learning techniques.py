
# coding: utf-8

# In[ ]:

import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
#from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn import svm
import re
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[ ]:

totaldata = pd.read_csv("/media/manashree/Study/SSA/data.txt")
totaldata.head()


# In[ ]:

def preprocess():
    global data
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)

    return tfidf_data


# In[ ]:

totaldata['overall'] = totaldata['overall'].map({1: 'negative',2: 'negative',3: 'neutral',4: 'positive',5: 'positive'})
totaldata.head()


# In[ ]:

totaldata.columns = ["labels","text"]
totaldata.head()


# In[ ]:

data = totaldata["text"]
target = totaldata["labels"]


# In[ ]:

def createSplits():
    global data_train,target_train,validation_data,test_data,validation_target,test_target
    # preparing data for split validation. 80% training, 20% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.4,random_state=43)
    validation_data,test_data,validation_target,test_target = cross_validation.train_test_split(data_test,target_test,test_size=0.5,random_state=43)
    print(len(data_train))
    print(len(data_test))
    print(len(validation_data))
    print(len(test_data))


# In[ ]:

'''
createSplits()

newDF = pd.DataFrame()
newDF['labels'] = target_train
newDF['text'] = data_train
newDF.to_csv("/media/manashree/Study/SSA/finalSplits/train.csv", index=False)

newDF2 = pd.DataFrame()
newDF2['labels'] = validation_target
newDF2['text'] = validation_data
newDF2.to_csv("/media/manashree/Study/SSA/finalSplits/validation.csv", index=False)
len(newDF2)

newDF3 = pd.DataFrame()
newDF3['labels'] = test_target
newDF3['text'] = test_data
newDF3.to_csv("/media/manashree/Study/SSA/finalSplits/test.csv", index=False)
len(newDF3)
'''


# In[ ]:

train = pd.read_csv("/media/manashree/Study/SSA/finalSplits/train.csv")
train_data = train['text']
train_labels = train['labels']

validation = pd.read_csv("/media/manashree/Study/SSA/finalSplits/validation.csv")
validation_data = validation['text']
validation_labels = validation['labels']

data = totaldata
tf_idf = preprocess()


# In[ ]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def learnBModel(ip,label,tst,tst_label):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    X_train = vectorizer.fit_transform(ip.data)
    X_test = vectorizer.transform(tst.data)
    tfidf_train = TfidfTransformer(use_idf=False).fit_transform(X_train)
    tfidf_test = TfidfTransformer(use_idf=False).fit_transform(X_test)
    classifier = BernoulliNB().fit(tfidf_train,label)
    predicted_BModel = classifier.predict(tfidf_test)
    evaluate_model(tst_label,predicted_BModel)


# In[ ]:

def evaluate_model(target_true,target_predicted):
    print(classification_report(target_true,target_predicted))
    print("The accuracy score is",accuracy_score(target_true,target_predicted))


# In[ ]:

print("BernoulliNB Results")
print(learnBModel(data_train,target_train,validation_data,validation_target))


# In[ ]:

X_train = traindata['text']
y_train = traindata['labels']

X_test = testdata['text']
y_test = testdata['labels']


# In[ ]:

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review,"html.parser").get_text() 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    return( " ".join( words ))  

clean_train_reviews = []


# In[ ]:

print()"Cleaning and parsing the training set movie reviews...\n"
for rec in X_train:
    clean_train_reviews.append( review_to_words(rec) )
print "Creating the bag of words...\n"


# In[ ]:

vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,
                             max_features = 10000)


# In[ ]:

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print "Training the random forest..."


# In[ ]:

forest = RandomForestClassifier(n_estimators = 1000,max_depth = 50) 
forest = forest.fit( train_data_features, y_train )
clean_test_reviews = [] 
print "Cleaning and parsing the test set movie reviews...\n"
for rec in X_test:
    clean_test_reviews.append( review_to_words(rec) )


# In[ ]:

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)
print "Random Forest Classifier Results"
print "The accuracy score is {:.2%}".format(accuracy_score(y_test.values,result))
print classification_report(y_test.values,result)


# In[ ]:

svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(train_data_features, y_train)
prediction_linear = svm_linear.predict(test_data_features)
print "SVM Results"
print "The accuracy score is {:.2%}".format(accuracy_score(y_test.values,prediction_linear))
print classification_report(y_test.values,prediction_linear)


# In[ ]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train_data_features, y_train)
prediction_NB = gnb.predict(test_data_features)
print "Naive Bayes Results"
print "The accuracy score is {:.2%}".format(accuracy_score(y_test.values,prediction_NB))
print classification_report(y_test.values,prediction_NB)


# In[ ]:

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

rbm = BernoulliRBM(n_components = 200, n_iter = 40, learning_rate = 0.01,  verbose = True)
logistic = linear_model.LogisticRegression()
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
logistic.fit(train_data_features, y_train)
classifier.fit(train_data_features, y_train)

# Get predictions
print "The BernoulliRBM model:"
prediction_NN = classifier.predict(test_data_features)
print "The accuracy score is {:.2%}".format(accuracy_score(y_test.values,prediction_NN))
print classification_report(y_test.values,prediction_NN)

