__author__ = 'zhangye'
#classify sentences
dir = "abstracts2_sen/"
import os
from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack,coo_matrix
from sklearn.metrics import f1_score, precision_score,recall_score
from pymedtermino import *
from pymedtermino.snomedct import *
import re
from normalize import normalize
from geniatagger import *
import drugbank
tagger = GeniaTagger('/Users/zhangye/Documents/Study/UTAustin/study/causal_Statement/geniatagger-3.0.1/geniatagger')
#print tagger.parse('Twenty patients undergoing liver surgery were randomly assigned\
     #   to IPM with INTEGER MEASURE_UNIT (15IPM) or 30 (30IPM) minutes ischemic intervals.')
file_dict = {}
for file in os.listdir(dir):
    if(not file.endswith('.txt')): continue
    name_key = file.split('.')[0]
    cur = open(dir+file,'rb')
    content_value = []
    label_value = []
    for c in cur:
        #normalize strings
        c = c.decode('unicode-escape')
        c = c.encode('ascii','ignore')
        content_value.append(c.strip()[:-1])
        label_value.append(int(c.split('\t')[-1].strip()))
    file_dict[name_key] = (content_value,label_value)
file_names = file_dict.keys()
print str(len(file_names)) + " abstracts"
all_sentences = []
all_labels = []
position_feature =[]
len_sen = []
#map document index to sentence index in feature matrix
#key is the document index
#value is the list of sentence indices
#extract POS features
index_map = {}
fea_to_abs = {}   #key is the index in the feature matrix, and value is the (name_key, sentence index)
start_index = 0
drug = drugbank.Drugbank()

def pos_to_string(sentence):    #convert a sentence into a sequence of POS tagger
    result = tagger.parse(sentence)
    return ' '.join([r[2] for r in result])
pos = []
#all_len = []
prev_index = []   #index of previous sentence
next_index = []   #index of next sentence
num_sentence = 0
for i, name in enumerate(file_names):
    print name
    #temp_sen = [drug_find.sub(f) for f in file_dict[name][0]]
    #pos  += [pos_to_string(j) for j in file_dict[name][0]]     #extract POS features using GENIA tagger
    #temp_pos,temp_len = zip(*temp)
    #all_len.extend(list(temp_len))
    all_sentences += [normalize(t,drug) for t in file_dict[name][0]]   #normalize each sentence
    all_labels += file_dict[name][1]
    for j in range(len(file_dict[name][0])): position_feature.append(j)    #extract position features
    index_map[i] = range(start_index,len(file_dict[name][0])+start_index)
    for k in range(start_index,len(file_dict[name][0])+start_index):fea_to_abs[k] = (name,k-start_index)
    '''
    for k,j in enumerate(index_map[i]):
        if(k==0):
            prev_index.append(10000)    #the first sentence in the abstract
            next_index.append(j+1)
        elif(k==len(index_map[i])-1):
            next_index.append(-10000)
            prev_index.append(j-1)
        else:
            prev_index.append(j - 1)
            next_index.append(j + 1)
    '''
    start_index += len(file_dict[name][0])
    num_sentence += len(file_dict[name][0])

#k = 1
#extract content features of previous k sentences and next k sentences
#k = 2
all_prev = []
all_next = []
all_len = []
all_prev_prev = []
all_next_next = []
for i, name in enumerate(file_names):
    sentences = file_dict[name][0]
    num_sen = len(sentences)
    #print name
    for j in range(num_sen):
        if(j==0):
            #print name
            all_prev.append("none")
            all_next.append(sentences[j+1])
            all_prev_prev.append("none")
            all_next_next.append(sentences[j+2])
        elif(j==num_sen-1):
            all_next.append("none")
            all_prev.append(sentences[j-1])
            all_next_next.append("none")
            all_prev_prev.append(sentences[j-2])
        elif(j==1):
            all_prev.append(sentences[0])
            all_next.append(sentences[2])
            all_prev_prev.append("none")
            all_next_next.append(sentences[3])
        elif(j==num_sen-2):
            all_next.append(sentences[j+1])
            all_prev.append(sentences[j-1])
            all_prev_prev.append(sentences[j-2])
            all_next_next.append("none")
        else:
            all_next.append(sentences[j+1])
            all_prev.append(sentences[j-1])
            all_next_next.append(sentences[j+2])
            all_prev_prev.append(sentences[j-2])
        if(len(sentences[j])>500): print "sentence too long:" + name + ":" + str(len(sentences[j]))

def tokenizer(input):
    return [t[0] for t in tagger.parse(input)]
count_vect = HashingVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b')
all_labels = np.array(all_labels)
count_pos = HashingVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b')
print "count vectorizer..."
X_train_counts = count_vect.fit_transform(all_sentences)
X_train_prev = count_vect.fit_transform(all_next)
X_train_next = count_vect.fit_transform(all_prev)
#X_train_next_next = count_vect.fit_transform(all_next_next)
#X_train_prev_prev = count_vect.fit_transform(all_prev_prev)
print "count vectorizer finish..."
#X_pos = count_pos.fit_transform(pos)
def doc_to_sen(docIndex,index_map):   #map document indices to sentence indices
    sentence_index = []
    for i in docIndex:
        sentence_index += index_map[i]
    return sentence_index

#combine features
features = hstack([X_train_counts,np.array(position_feature).reshape(-1,1)])
features = hstack([features,X_train_next])
features = hstack([features,X_train_prev])
#features = hstack([features,np.array(all_len).reshape(-1,1)])
#features = hstack([features,X_pos])
#features = hstack([features,X_train_next_next])
#features = hstack([features,X_train_prev_prev])
params = {"alpha":(0.0001,0.001,0.01,0.1,1,10,100)}
#scaler1 = StandardScaler(with_mean=False)
#scaler2 = StandardScaler(with_mean=False)
for p in params["alpha"]:
    #generate CV folds
    clf = SGDClassifier(loss="hinge",penalty="l2",n_iter=250,alpha=p)
    clf1 = SGDClassifier(loss="hinge",penalty="l2",n_iter=250,alpha=p)
    kf = KFold(len(file_names),n_folds=5,shuffle=True)
    f1s = []
    pres = []
    recall = []
    FP = open('fp.txt','wb')
    FN = open('fn.txt','wb')
    for tr_doc,te_doc in kf:
        train_index = doc_to_sen(tr_doc,index_map)
        test_index = doc_to_sen(te_doc,index_map)
        train_data = features.tocsr()[train_index,:]
        train_label = all_labels[train_index]
        test_data = features.tocsr()[test_index,:]
        test_label = all_labels[test_index]
        #train_data = scaler1.fit_transform(train_data)
        clf.fit(train_data,train_label)

        sorted_index_train = []     #the sorted index in the absract
        for t in tr_doc:
            #current_scores = []
            sen_index = doc_to_sen([t],index_map)
            cur_train_score = clf.decision_function(features.tocsr()[sen_index,:])
            #obtain the sorted position of each sentence according to the distance to the boundary
            temp = [i[0] for i in sorted(enumerate(list(cur_train_score)),key=lambda x:x[1],\
                                                 reverse=True)]
            sorted_index = np.zeros(len(temp))
            for i, q in enumerate(temp): sorted_index[q] = i
            sorted_index_train += list(sorted_index)

        #add the previous max score to train new SVM
        train_data = hstack([train_data,np.array(sorted_index_train).reshape(-1,1)])
        #train_data = scaler2.fit_transform(train_data)
        #clf1.fit(train_data,train_label)
        #test_data = scaler1.transform(test_data)
        test_score = clf.decision_function(test_data)
        sorted_index_test = []
        prediction = []
        for t in te_doc:
            sen_index = doc_to_sen([t],index_map)
            cur_test_score = clf.decision_function(features.tocsr()[sen_index,:])
            temp = [i[0] for i in sorted(enumerate(list(cur_test_score)),key=lambda x:x[1],reverse=True)]
            sorted_index = np.zeros(len(temp))
            for i, q in enumerate(temp): sorted_index[q] = i
            sorted_index_test += list(sorted_index)
            pred_abs = np.zeros(len(temp))
            pred_abs[temp[0]] = 1
            prediction += list(pred_abs)
        #test_data = hstack([test_data,np.array(sorted_index_test).reshape(-1,1)])
        #test_data = scaler2.transform(test_data)
        #prediction = clf1.predict(test_data)
        #write false positive and false negative instance into a file       #
        abs_sen = [fea_to_abs[t] for t in np.array(test_index)[np.logical_and(prediction==1,test_label==0)]]
        map(lambda x: FP.write(x[0]+"\t"+file_dict[x[0]][0][x[1]]+"\n"), abs_sen)
        abs_sen_1 = [fea_to_abs[t] for t in np.array(test_index)[np.logical_and(prediction==0,test_label==1)]]
        map(lambda x: FN.write(x[0]+"\t"+file_dict[x[0]][0][x[1]]+"\n"), abs_sen_1)
        f1 = f1_score(test_label,prediction)
        f1s.append(f1)
        pres.append(precision_score(test_label,prediction))
        recall.append(recall_score(test_label,prediction))
    print "alpha= "+str(p)
    print "mean F1 score " + str(sum(f1s)/len(f1s))
    print "mean precision score " + str(sum(pres)/len(pres))
    print "mean recall " + str(sum(recall)/len(pres))
    FP.close()
    FN.close()




