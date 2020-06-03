import pickle
import nltk
from nltk.util import ngrams
import heapq 
import numpy as np
import string
import sys
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2
import re 

#**************************************FOR FEATURE 1&2*************************************

feature_len = int(sys.argv[2])
RANDOM_STATE = int(sys.argv[3])

lines = []
with open('../Dataset/dataset_humour_processed.pkl', 'rb') as fp:
    lines_1 = pickle.load(fp)
for k in lines_1.keys():
    lines.append(lines_1[k])

#'''
dataset = [(' ').join([j[:-3] for j in i['tweet']]) for i in lines]
word2count = {} 
for data in dataset: 
    words = nltk.word_tokenize(data) 
    for word in words: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
print(len(word2count))
freq_words = heapq.nlargest(feature_len, word2count, key=word2count.get)
X_bow = [] 
for num, data in enumerate(dataset): 
    print(num)
    vector = [] 
    for word in freq_words: 
        if word in nltk.word_tokenize(data): 
            vector.append(1) 
        else: 
            vector.append(0) 
    X_bow.append(vector) 
X_bow = np.asarray(X_bow) 

data_trigram = []
n = 3
data_trigram = [[b[i:i+n] for i in range(len(b)-n+1)] for b in dataset]
word2count = {} 
for data in data_trigram:  
    for word in data: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
print(len(word2count))
freq_words = heapq.nlargest(feature_len, word2count, key=word2count.get)
X_tri = [] 
for num, data in enumerate(data_trigram): 
    print(num)
    vector = [] 
    for word in freq_words: 
        if word in data: 
            vector.append(1) 
        else: 
            vector.append(0) 
    X_tri.append(vector) 
X_tri = np.asarray(X_tri) 

#**************************************FOR FEATURE 3*************************************

def camel_case_split(str): 
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str) 
with open('../Dataset/data_with_tweetids.pkl', 'rb') as fp:
    orignal = pickle.load(fp)
punctuation =  string.punctuation  
tweet_id = [int(i[0]) for i in orignal]
orignal = [[j[0] for j in i[1:-1] if j[0] not in punctuation] for i in orignal]
orignal_1 = []
for i in orignal:
	temp =[]
	for j in i:
		flag = 0
		for k in j:
			if k in punctuation:
				flag = 1
				break
		if j[0]=='#':
			flag=2
		if flag==0:
			temp.append(j)
		elif flag==1:
			for l in j.split(k):
				temp.append(l)
		elif flag==2:
			if len(camel_case_split(j))==0:
				temp.append(j[1:])
			else:
				for l in camel_case_split(j):
					temp.append(l)
	temp = [i.lower() for i in temp if i!='']
	orignal_1.append(temp)
orignal_dict = {}
for i,j in zip(tweet_id, orignal_1):
	orignal_dict[i] = j
orignal = []
for i in lines_1.keys():
	orignal.append(orignal_dict[i])

dataset_hash = [(' ').join(i) for i in orignal]
word2count = {} 
for data in dataset_hash: 
    words = nltk.word_tokenize(data) 
    for word in words: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
print(len(word2count))
freq_words = heapq.nlargest(feature_len, word2count, key=word2count.get)
X_hashtag = [] 
for num, data in enumerate(dataset_hash):
    print(num)
    vector = [] 
    for word in freq_words: 
        if word in nltk.word_tokenize(data): 
            vector.append(1) 
        else: 
            vector.append(0) 
    X_hashtag.append(vector) 
X_hashtag = np.asarray(X_hashtag) 

#*********************MODEL********************************************

X = np.hstack((X_bow, X_tri, X_hashtag))
Y = np.array([i['label'] for i in lines])
X = SelectKBest(chi2, k=feature_len).fit_transform(X, Y)

result = []
def train(X_train, X_test, y_train, y_test):
    clf = SVC(kernel='linear', random_state = RANDOM_STATE)
    clf.fit(X_train, y_train)
    print('train: ', clf.score(X_train, y_train), 'test: ', clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i, j in zip(y_pred,y_test):
        if i==0 and j ==0 :
            tn = tn + 1
        elif i==1 and j ==1 :
            tp = tp + 1
        elif i==0 and j ==1 :
            fn = fn + 1
        elif i==1 and j ==0 :
            fp = fp + 1
    P = float(tp)/(tp+fp)
    R = float(tp)/(tp+fn)
    F1 = float(2*P*R)/ (P+R)
    A = float(tp+tn)/(tp+tn+fp+fn)
    print ("Acc",A*100 , "F1",F1*100)
    result.append([A*100, F1*100])

kf = KFold(n_splits=10)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    train(X_train, X_test, y_train, y_test)

print(result)
print('Acc: ', sum([i[0] for i in result])/10, 'F1: ', sum([i[1] for i in result])/10)
with open('results_baseline.csv', 'a') as f:
    f.write(sys.argv[1] + ' Acc: ' + str(sum([i[0] for i in result])/10) + ' F1: ' + str(sum([i[1] for i in result])/10))
