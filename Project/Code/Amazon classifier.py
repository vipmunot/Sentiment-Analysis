
# coding: utf-8

# In[1]:

import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn import svm
import re
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


with open('elect.json', 'rb') as f:
    data = f.readlines()
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"

review = pd.read_json(data_json_str)


# In[2]:

label = []
for entry in review['overall']:
    if entry in (1,2):
        label.append('negative')
    elif entry in (4,5):
        label.append('positive')
    else:
        label.append('neutral')
review.insert( 2,'label', label)
review = review[['reviewText','label']]


# In[3]:

review.head()


# In[4]:

data = review['reviewText']
target = review['label']
# In[5]:

def preprocess():
    global data
    global target
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)

    return tfidf_data


# In[6]:

def learn_model(data,target):
    # preparing data for split validation. 80% training, 20% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.2,random_state=43)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    evaluate_model(target_test,predicted)


# In[7]:

def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))


# In[8]:

tf_idf = preprocess()
print "BernoulliNB Results"
print learn_model(tf_idf,target)


# In[9]:

X_train, X_test, y_train, y_test = train_test_split(review['reviewText'], review['label'], test_size=0.20, random_state=4212)

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review,"html.parser").get_text() 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    return( " ".join( words ))  

clean_train_reviews = []


# In[10]:

print "Cleaning and parsing the training set movie reviews...\n"
for rec in X_train:
    clean_train_reviews.append( review_to_words(rec) )
print "Creating the bag of words...\n"


# In[11]:

vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 10000) 


# In[12]:

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print "Training the random forest..."


# In[13]:

forest = RandomForestClassifier(n_estimators = 1000,max_depth = 50) 
forest = forest.fit( train_data_features, y_train )
clean_test_reviews = [] 
print "Cleaning and parsing the test set movie reviews...\n"
for rec in X_test:
    clean_test_reviews.append( review_to_words(rec) )


# In[17]:

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)
print "Random Forest Classifier Results"
print "The accuracy score is {:.2%}".format(accuracy_score(y_test.values,result))
print classification_report(y_test.values,result)


# In[18]:

svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(train_data_features, y_train)
prediction_linear = svm_linear.predict(test_data_features)
print "SVM Results"
print "The accuracy score is {:.2%}".format(accuracy_score(y_test.values,prediction_linear))
print classification_report(y_test.values,prediction_linear)


# In[20]:

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


# In[ ]:



