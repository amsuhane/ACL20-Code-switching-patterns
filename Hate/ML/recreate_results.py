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

feature_len = int(sys.argv[1])
RANDOM_STATE = int(sys.argv[2])

lines = []
with open('../Dataset/dataset_hate_processed_manual_annotated.pkl', 'rb') as fp:
    lines_1 = pickle.load(fp)
for k in lines_1.keys():
    lines.append(lines_1[k])

dataset = [(' ').join([j[:-3] for j in i['tweet']]) for i in lines]

#**************************************FOR WORD N-GRAMS*************************************

word2count = {} 
for data in dataset: 
    words = nltk.word_tokenize(data) 
    for word in words: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
freq_words = heapq.nlargest(sum(np.array([i[1] for i in word2count.items()])>10), word2count, key=word2count.get)
print('W1: ', len(word2count), len(freq_words))
X_bow = [] 
for num, data in enumerate(dataset): 
    #print(num)
    vector = [] 
    for word in freq_words: 
        if word in nltk.word_tokenize(data): 
            vector.append(1) 
        else: 
            vector.append(0) 
    X_bow.append(vector) 
X_bow = np.asarray(X_bow) 

data_bigram = []
for text in dataset:
    token=nltk.word_tokenize(text)
    bigrams=ngrams(token,2)
    data_bigram.append([(' ').join(i) for i in bigrams])    
word2count = {} 
for data in data_bigram:  
    for word in data: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
freq_words = heapq.nlargest(sum(np.array([i[1] for i in word2count.items()])>10), word2count, key=word2count.get)
print('W2: ', len(word2count), len(freq_words))
X_bi = [] 
for num, data in enumerate(data_bigram): 
    #print(num)
    vector = [] 
    for word in freq_words: 
        if word in data: 
            vector.append(1) 
        else: 
            vector.append(0) 
    X_bi.append(vector) 
X_bi = np.asarray(X_bi) 


data_trigram = []
for text in dataset:
    token=nltk.word_tokenize(text)
    trigrams=ngrams(token,3)
    data_trigram.append([(' ').join(i) for i in trigrams])    
word2count = {} 
for data in data_trigram:  
    for word in data: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
freq_words = heapq.nlargest(sum(np.array([i[1] for i in word2count.items()])>10), word2count, key=word2count.get)
print('W3: ', len(word2count), len(freq_words))
X_tri = [] 
for num, data in enumerate(data_trigram): 
    #print(num)
    vector = [] 
    for word in freq_words: 
        if word in data: 
            vector.append(1) 
        else: 
            vector.append(0) 
    X_tri.append(vector) 
X_tri = np.asarray(X_tri) 

#**************************************FOR LETTER N-GRAMS*************************************

data_letter_unigram = []
n = 1
data_letter_unigram = [[b[i:i+n] for i in range(len(b)-n+1)] for b in dataset]
word2count = {} 
for data in data_letter_unigram:  
    for word in data: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
freq_words = heapq.nlargest(sum(np.array([i[1] for i in word2count.items()])>8), word2count, key=word2count.get)
print('C1: ', len(word2count), len(freq_words))
X_letter_uni = [] 
for num, data in enumerate(data_letter_unigram): 
    #print(num)
    vector = [] 
    for word in freq_words: 
        if word in data: 
            vector.append(1) 
        else: 
            vector.append(0) 
    X_letter_uni.append(vector) 
X_letter_uni = np.asarray(X_letter_uni) 

data_letter_bigram = []
n = 2
data_letter_bigram = [[b[i:i+n] for i in range(len(b)-n+1)] for b in dataset]
word2count = {} 
for data in data_letter_bigram:  
    for word in data: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
freq_words = heapq.nlargest(sum(np.array([i[1] for i in word2count.items()])>8), word2count, key=word2count.get)
print('C2: ', len(word2count), len(freq_words))
X_letter_bi = [] 
for num, data in enumerate(data_letter_bigram): 
    #print(num)
    vector = [] 
    for word in freq_words: 
        if word in data: 
            vector.append(1) 
        else: 
            vector.append(0) 
    X_letter_bi.append(vector) 
X_letter_bi = np.asarray(X_letter_bi) 

data_letter_trigram = []
n = 1
data_letter_trigram = [[b[i:i+n] for i in range(len(b)-n+1)] for b in dataset]
word2count = {} 
for data in data_letter_trigram:  
    for word in data: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
freq_words = heapq.nlargest(sum(np.array([i[1] for i in word2count.items()])>8), word2count, key=word2count.get)
print('C3: ', len(word2count), len(freq_words))
X_letter_tri = [] 
for num, data in enumerate(data_letter_trigram): 
    print(num)
    vector = [] 
    for word in freq_words: 
        if word in data: 
            vector.append(1) 
        else: 
            vector.append(0) 
    X_letter_tri.append(vector) 
X_letter_tri = np.asarray(X_letter_tri) 

#----------------------------------- Lexicon -----------------------------------------
with open('../Dataset/hate_lexicon.txt', 'rb') as F:
    hate_lexicon = F.readlines()
hate_lexicon = [str(i)[2:-3] for i in hate_lexicon]

X_lexicon = []
count = 0
for i in dataset:
    for j in i.split(' '):
        if j in hate_lexicon:
            count+=1
    X_lexicon.append(count)
    count = 0


#*********************MODEL********************************************


def train(X_train, X_test, y_train, y_test):
    clf = SVC(kernel='rbf', C=10, random_state = RANDOM_STATE)
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

def featureselection(features, train_tweets, train_truth):
	model = SelectKBest(score_func=chi2, k=feature_len)
	fit = model.fit(np.array(train_tweets), np.array(train_truth))
	return fit.transform(np.array(features)).tolist()

X = np.hstack((X_bow, X_bi, X_tri, X_letter_uni, X_letter_bi, X_tri))
X_switch = np.hstack((X_bow, X_bi, X_tri, X_letter_uni, X_letter_bi, X_tri, X_signal_switching))
label_to_int = {'Hate':1, 'Normal':0}
Y = np.array([label_to_int[i['label']] for i in lines])
print(X.shape, X_switch.shape)

#-----------------------------WITH LEAKAGE---------------------------#

result = []
X_leakage = SelectKBest(chi2, k=1200).fit_transform(X,Y)
kf = KFold(n_splits=10)
kf.get_n_splits(X_leakage)
for train_index, test_index in kf.split(X_leakage):
    X_train, X_test = X_leakage[train_index], X_leakage[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    train(X_train, X_test, y_train, y_test)

print("With leakage, baseline: ")
print('Acc: ', sum([i[0] for i in result])/10, 'F1: ', sum([i[1] for i in result])/10, '\n')
print(len(result))

with open('hate.csv', 'a') as f:
	f.write('baseline'+'\t'+'Yes'+'\t'+sys.argv[1]+'\t'+sys.argv[2]+'\t'+str(sum([i[0] for i in result])/10)+'\t'+ str(sum([i[1] for i in result])/10)+'\n')

#-----------------------------NO LEAKAGE---------------------------#

result = []
kf = KFold(n_splits=10)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    X_test = featureselection(X_test, X_train, y_train)
    X_train = featureselection(X_train, X_train, y_train)
    train(X_train, X_test, y_train, y_test)

print("No leakage, baseline: ")
print('Acc: ', sum([i[0] for i in result])/10, 'F1: ', sum([i[1] for i in result])/10, '\n')

with open('hate.csv', 'a') as f:
	f.write('baseline'+'\t'+'No'+'\t'+sys.argv[1]+'\t'+sys.argv[2]+'\t'+str(sum([i[0] for i in result])/10)+'\t'+ str(sum([i[1] for i in result])/10)+'\n')

#---------------------------WITH SWITCHING NO LEAKAGE ---------------#

result = []
kf = KFold(n_splits=10)
kf.get_n_splits(X_switch)
for train_index, test_index in kf.split(X_switch):
    X_train, X_test = X_switch[train_index], X_switch[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    X_test = featureselection(X_test, X_train, y_train)
    X_train = featureselection(X_train, X_train, y_train)    
    train(X_train, X_test, y_train, y_test)

print("No leakage, switching: ")
print('Acc: ', sum([i[0] for i in result])/10, 'F1: ', sum([i[1] for i in result])/10, '\n')
print(len(result))

with open('hate.csv', 'a') as f:
	f.write('Switching'+'\t'+'No'+'\t'+sys.argv[1]+'\t'+sys.argv[2]+'\t'+str(sum([i[0] for i in result])/10)+'\t'+ str(sum([i[1] for i in result])/10)+'\n')

#---------------------------WITH SWITCHING WITH LEAKAGE ---------------#

result = []
X_leakage = SelectKBest(chi2, k=1200).fit_transform(X_switch,Y)
kf = KFold(n_splits=10)
kf.get_n_splits(X_leakage)
for train_index, test_index in kf.split(X_leakage):
    X_train, X_test = X_leakage[train_index], X_leakage[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    train(X_train, X_test, y_train, y_test)

print("With leakage: ")
print('Acc: ', sum([i[0] for i in result])/10, 'F1: ', sum([i[1] for i in result])/10, '\n')
print(len(result))

with open('hate.csv', 'a') as f:
	f.write('Switching'+'\t'+'Yes'+'\t'+sys.argv[1]+'\t'+sys.argv[2]+'\t'+str(sum([i[0] for i in result])/10)+'\t'+ str(sum([i[1] for i in result])/10)+'\n')
