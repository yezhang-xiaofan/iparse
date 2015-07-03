__author__ = 'zhangye'
#classify sentences
dir = "abstracts2_sen/"
import os
from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.svm import SVC
from scipy.sparse import hstack,coo_matrix
from sklearn.metrics import f1_score, precision_score,recall_score
from pymedtermino import *
from pymedtermino.snomedct import *
import re
from geniatagger import *
tagger = GeniaTagger('/Users/zhangye/Documents/Study/UTAustin/study/causal_Statement/geniatagger-3.0.1/geniatagger')
print tagger.parse('This is a pen.')

concept = SNOMEDCT[302509004]
SNOMEDCT.has_concept(302509004)
file_dict = {}
for file in os.listdir(dir):
    if(not file.endswith('.txt')): continue
    name_key = file.split('.')[0]
    cur = open(dir+file,'rb')
    content_value = []
    label_value = []
    for c in cur:
        c = c.decode('ascii',errors = 'ignore')
        content_value.append(c.strip()[:-1])
        label_value.append(int(c.split('\t')[-1].strip()))
    file_dict[name_key] = (content_value,label_value)
file_names = file_dict.keys()
all_sentences = []
all_labels = []
position_feature =[]
len_sen = []
#map document index to sentence index in feature matrix
#key is the document index
#value is the list of sentence indices
#extract POS features
index_map = {}
start_index = 0

for i, name in enumerate(file_names):
    all_sentences += file_dict[name][0]
    all_labels += file_dict[name][1]
    for j in range(len(file_dict[name][0])):
        position_feature.append(j)
    index_map[i] = range(start_index,len(file_dict[name][0])+start_index)
    start_index += len(file_dict[name][0])

#k = 1
#extract content features of previous k sentences and next k sentences
all_prev = []
all_next = []
all_len = []
pos = []
def pos_to_string(sentence):    #convert a sentence into a sequence of POS tagger
    result = tagger.parse(sentence)
    return ' '.join([r[2] for r in result])

for i, name in enumerate(file_names):
    sentences = file_dict[name][0]
    num_sen = len(sentences)
    for j in range(num_sen):
        if(j==0):
            print name
            all_prev.append("none")
            all_next.append(sentences[j+1])
        elif(j==num_sen-1):
            all_next.append("none")
            all_prev.append(sentences[j-1])
        else:
            all_next.append(sentences[j+1])
            all_prev.append(sentences[j-1])
        all_len.append(len(re.findall(r'\w+',sentences[j])))
        pos.append(pos_to_string(sentences[j]))

count_vect = CountVectorizer(stop_words='english',ngram_range=(1,2))
all_labels = np.array(all_labels)
count_pos = CountVectorizer(ngram_range=(1,3))
X_train_counts = count_vect.fit_transform(all_sentences)
X_train_prev = count_vect.fit_transform(all_next)
X_train_next = count_vect.fit_transform(all_prev)
X_pos = count_pos.fit_transform(pos)
def doc_to_sen(docIndex,index_map):
    sentence_index = []
    for i in docIndex:
        sentence_index += index_map[i]
    return sentence_index

#combine features
features = hstack([X_train_counts,np.array(position_feature).reshape(-1,1)])
features = hstack([features,X_train_next])
features = hstack([features,X_train_prev])
features = hstack([features,np.array(all_len).reshape(-1,1)])
features = hstack([features,X_pos])
params = {"alpha":(0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000)}
for p in params["alpha"]:
    #generate CV folds
    clf = SGDClassifier(n_iter=250,alpha=p)
    kf = KFold(len(file_names),n_folds=5,shuffle=True)
    f1s = []
    pres = []
    recall = []
    for tr_doc,te_doc in kf:
        train_index = doc_to_sen(tr_doc,index_map)
        test_index = doc_to_sen(te_doc,index_map)
        train_data = features.tocsr()[train_index,:]
        train_label = all_labels[train_index]
        test_data = features.tocsr()[test_index,:]
        test_label = all_labels[test_index]
        clf.fit(train_data,train_label)
        prediction = clf.predict(test_data)
        f1 = f1_score(test_label,prediction)
        f1s.append(f1)
        pres.append(precision_score(test_label,prediction))
        recall.append(recall_score(test_label,prediction))
    print "alpha= "+str(p)
    print "mean F1 score " + str(sum(f1s)/len(f1s))
    print "mean precision score " + str(sum(pres)/len(pres))
    print "mean recall " + str(sum(recall)/len(pres))
#gs_clf = GridSearchCV(clf,param_grid=params,n_jobs=-1,scoring='f1',cv=cv)
#gs_clf.fit(features,np.array(all_labels).astype(int))
#print gs_clf.grid_scores_
#clf.fit(X_train_counts,all_labels)
#predicton = clf.predict(X_train_counts)
#print classification_report(all_labels,predicton)





