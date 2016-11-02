
# coding: utf-8

                
# In[1]:

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score

import timeit
from datetime import datetime


# In[2]:

stime = datetime.now()
print datetime.now()


# In[3]:

with open('/home/alien/Desktop/lbl_review_test.json', 'rb') as f:
    data = f.readlines()


# In[4]:

data = map(lambda x: x.rstrip(), data)

data_json_str = "[" + ','.join(data) + "]"

pos = pd.read_json(data_json_str)


# In[5]:

print len(pos)


# In[6]:

data = pos['text']
target = pos['Sentiment']
stars = pos['stars']


# In[7]:

def preprocess():
    global data
    global target
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)

    return tfidf_data


# In[8]:

def learn_model(data,target):
    # preparing data for split validation. 80% training, 20% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.2,random_state=43)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    evaluate_model(target_test,predicted)


# In[9]:

def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))


# In[10]:

tf_idf = preprocess()
learn_model(tf_idf,target)
print learn_model(tf_idf,target)

# In[11]:

etime = datetime.now()
print datetime.now()
print etime - stime



