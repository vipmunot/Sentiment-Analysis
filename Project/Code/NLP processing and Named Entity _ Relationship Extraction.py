
# coding: utf-8

# In[ ]:

import pandas as pd
import os
import re
import string
import nltk

default_path= "/media/manashree/Study/SSA/"
os.chdir(default_path)

'''
with open('reviews_Electronics_10.json', 'r') as f:
    data = f.readlines()
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"
    
data = pd.read_json(data_json_str)
'''
data = pd.read_csv("train-small.csv")


# In[ ]:

def ie_process(document):
    "returns named entity chunks in a given text"
    sentences = nltk.sent_tokenize(document)
    tokenized = [nltk.word_tokenize(sentence.translate(string.punctuation)) for sentence in sentences]
    pos_tags  = [nltk.pos_tag(sentence) for sentence in tokenized]
    #print(pos_tags)
    chunked_sents = nltk.ne_chunk_sents(pos_tags, binary=True)
    return chunked_sents

def review_to_words(raw_review):
    # 1. Remove HTML
    #review_text = BeautifulSoup(raw_review).get_text() 
    review_text = raw_review
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    stops.discard('not')
    stops.discard('nor')
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join(words))

def find_entities(chunks):
    "given list of tagged parts of speech, returns unique named entities"

    def traverse(tree):
        "recursively traverses an nltk.tree.Tree to find named entities"
          
        entity_names = []
        
        if hasattr(tree, 'label') and tree.label:
            if tree.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in tree]))
            else:
                for child in tree:
                    entity_names.extend(traverse(child))
    
        return entity_names
    
    named_entities = []
    
    for chunk in chunks:
        entities = sorted(list(set([word for tree in chunk
                            for word in traverse(tree)])))
        for e in entities:
            if e not in named_entities:
                named_entities.append(e)
    return named_entities

def extract_relationships(chnks):    
    vnv = """(
        is/V|    # 3rd sing present and
        was/V|   # past forms of the verb zijn ('be')
    )
    *       # followed by anything
    van/Prep # followed by van ('of')
    """
    
    IN = re.compile(r'.*\bin\b(?!\b.+ing)')
    VAN = re.compile(vnv, re.VERBOSE)
    for doc in chnks:
        for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern = IN):
            print(nltk.sem.rtuple(rel))
        for r in nltk.sem.extract_rels('PER', 'ORG', doc,corpus='conll2002', pattern=VAN):
            print(nltk.sem.clause(r, relsym="VAN"))


# In[ ]:

from nltk.corpus import conll2002

#print(list(data.columns.values))
#data2 = data.ix[:,'reviewText']
data2 = data['text']
#f2 = open('/media/manashree/Study/SSA/Proj/amazon-fine-foods/tagged.txt','w')
#print(data2)

for row in data2:
    print(row)
    tree = treeCopy = ie_process(row)
    named_entities2 = find_entities(tree)
    for t in named_entities2:
        print(" entities: ",t)   
    #print("Relationships")
    ne2 = extract_relationships(treeCopy)


# In[ ]:

from nltk.sem.drt import AnaphoraResolutionException

import nltk
from nltk.sem.logic import *

print('The nltk version is {}.'.format(nltk.__version__))

print(resolve_anaphora(chunks))

