
# coding: utf-8

# In[1]:

import pandas as pd
import re
from bs4 import BeautifulSoup
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime


# In[2]:

stime = datetime.now()
print datetime.now()


# In[3]:

with open('lbl_review_test.json', 'rb') as f:
    data = f.readlines()


# In[4]:

data = map(lambda x: x.rstrip(), data)
data_json_str = "[" + ','.join(data) + "]"
data_df = pd.read_json(data_json_str)


# In[5]:

new_data = data_df
print len(new_data)

# In[6]:

X_train, X_test, y_train, y_test = train_test_split(new_data["text"], new_data["Sentiment"], test_size=0.20, random_state=4212)

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review,'lxml').get_text() 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    return( " ".join( words ))  

clean_train_reviews = []


# In[7]:

print "Cleaning and parsing the training set movie reviews...\n"
for rec in X_train:
    clean_train_reviews.append( review_to_words(rec) )
print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit( train_data_features, y_train )
clean_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for rec in X_test:
    clean_test_reviews.append( review_to_words(rec) )


# In[8]:

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"emotion":result,"label":y_test} )
output.to_csv( "Bag_of_Words_model_test.csv", index=False, quoting=csv.QUOTE_NONE, quotechar='')


# In[ ]:

accuracy_score(y_test.values,result)
print "accuracy_score: ", accuracy_score(y_test.values,result)

# In[ ]:

from sklearn import svm
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(train_data_features, y_train)
prediction_linear = svm_linear.predict(test_data_features)
print prediction_linear

# In[ ]:

etime = datetime.now()
print datetime.now()
print etime - stime





