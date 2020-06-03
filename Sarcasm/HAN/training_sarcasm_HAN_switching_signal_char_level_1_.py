SEED=42
from numpy.random import seed
seed(SEED)
from tensorflow import set_random_seed
set_random_seed(SEED)

import numpy as np
import pandas as pd
import sys
from collections import defaultdict
import re
import random
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, concatenate,LSTM

from keras.engine.topology import Layer, InputSpec
from keras import initializers
import pickle
import sys
import os
from keras.models import Model,Sequential
from keras.layers import Dense, Embedding, Input
from keras.preprocessing import text as keras_text, sequence as keras_seq
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split



def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


with open('../Datasets/embedding_50d.pkl','rb') as F:
  embedding=pickle.load(F)#, encoding="bytes")

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
      

def init_embedding():
    embedding['oov']=[random.random() for i in range(50)]


with open(sys.argv[7],'rb') as F:
  data1=pickle.load(F)

init_embedding()

list_sentences_train=[]
y=[]
labels=[]
signal_feature=[]
# for key in data1.keys():
#     z=[]
#     data=data1[key]['tweet']
#     label=data1[key]['label']
#     cnt_hin,cnt_en=0,0
#     v=0
#     f1,f2=0,0
#     tot_cntt=0
#     for j in range(len(data)):

#       if data[j].split("_")[1]=="en":
#         f1+=1
#       elif data[j].split("_")[1]=="hi":
#         f2+=1
#       tot_cntt+=1
      
#       if sys.argv[1]=="EnHi":
#         if j>0 and data[j-1].split("_")[1]=="en" and data[j].split("_")[1]=="hi":
#           cnt_hin+=1
        
#         if j>0 and data[j-1].split("_")[1]=="hi" and data[j].split("_")[1]=="en":
#           cnt_en-=1
#       elif sys.argv[1]=="Combined" or sys.argv[1]=="Fraction":
#         if j>0 and ((data[j-1].split("_")[1]=="en" and data[j].split("_")[1]=="hi") or 
#                            (data[j-1].split("_")[1]=="hi" and data[j].split("_")[1]=="en")):
#           v+=1

    
#     f1,f2=f1/(tot_cntt*1.00),f2/(tot_cntt*1.00)
#     #print(f1,f2)
#     labels.append(label)

#     # Appendending the data    
#     list_sentences_train.append(data)

#     pre_hi=[0]*(len(data)+1)
#     pre_en=[0]*(len(data)+1)

#     val_arr_enhi=[0]*(len(data)+1)
#     val_arr_hien=[0]*(len(data)+1)
#     for i in range(len(data)):
#       pre_hi[i+1] = (data[i].split("_")[1]=='hi') + pre_hi[i]
#       pre_en[i+1] = (data[i].split("_")[1]=='en') + pre_en[i]

#     for i in range(len(data)):
#       if data[i].split("_")[1]=='hi' :
#         val_arr_enhi[i] = pre_en[i+1]
#       if data[i].split("_")[1]=='en' :
#         val_arr_hien[i] = pre_hi[i+1]
    
#     mean_hi,stddev_hi=np.mean(val_arr_enhi),np.std(val_arr_enhi)
#     mean_en,stddev_en=np.mean(val_arr_hien),np.std(val_arr_hien)
#     #Switching mode combining
#     if sys.argv[1]=="EnHi":
#       #signal_feature.append([cnt_en,cnt_hin,mean_hi,stddev_hi,mean_en,stddev_en])
#       signal_feature.append([cnt_en,cnt_hin])
#     elif sys.argv[1]=="Combined":
#       #signal_feature.append([v,mean_hi,stddev_hi,mean_en,stddev_en])
#       signal_feature.append(v)
#     elif sys.argv[1]=="Fraction":
#       #signal_feature.append([f1,f2,v,mean_hi,stddev_hi,mean_en,stddev_en])
#       signal_feature.append([f1,f2,v])
#     elif sys.argv[1]=="All9Signals":
#       signal_feature.append([cnt_en,cnt_hin,v,f1,f2,mean_hi,stddev_hi,mean_en,stddev_en])

for key in data1.keys():
    z=[]
    data=data1[key]['tweet']
    label=data1[key]['label']
    #print(data)
    cnt_hin,cnt_en=0,0
    v=0
    f1,f2=0,0
    tot_cntt=0
    for j in range(len(data)):
      if len(data[j].split("_"))==1:
        continue
      
      if data[j].split("_")[1]=="en":
        f1+=1
      elif data[j].split("_")[1]=="hi":
        f2+=1
      tot_cntt+=1
      
      if j>0 and data[j-1].split("_")[1]=="en" and data[j].split("_")[1]=="hi":
        cnt_hin+=1
    
      if j>0 and data[j-1].split("_")[1]=="hi" and data[j].split("_")[1]=="en":
        cnt_en+=1

      if j>0 and ((data[j-1].split("_")[1]=="en" and data[j].split("_")[1]=="hi") or (data[j-1].split("_")[1]=="hi" and data[j].split("_")[1]=="en")):
        v+=1

    
    f1,f2=f1/(tot_cntt*1.00),f2/(tot_cntt*1.00)
    #print(f1,f2)
    labels.append(label)

    # Appendending the data    
    list_sentences_train.append(data)

    pre_hi=[0]*(len(data)+1)
    pre_en=[0]*(len(data)+1)

    val_arr_enhi=[0]*(len(data)+1)
    val_arr_hien=[0]*(len(data)+1)
    for i in range(len(data)):
      pre_hi[i+1] = (data[i].split("_")[1]=='hi') + pre_hi[i]
      pre_en[i+1] = (data[i].split("_")[1]=='en') + pre_en[i]

    for i in range(len(data)):
      if data[i].split("_")[1]=='hi' :
        val_arr_enhi[i] = pre_en[i+1]
      if data[i].split("_")[1]=='en' :
        val_arr_hien[i] = pre_hi[i+1]
    
    mean_hi,stddev_hi=np.mean(val_arr_enhi),np.std(val_arr_enhi)
    mean_en,stddev_en=np.mean(val_arr_hien),np.std(val_arr_hien)
    #Switching mode combining
    if sys.argv[1]=="EnHi":
      #signal_feature.append([cnt_en,cnt_hin,mean_hi,stddev_hi,mean_en,stddev_en])
      signal_feature.append([cnt_en,cnt_hin])
    elif sys.argv[1]=="Combined":
      #signal_feature.append([v,mean_hi,stddev_hi,mean_en,stddev_en])
      signal_feature.append(v)
    elif sys.argv[1]=="Fraction":
      #signal_feature.append([f1,f2,v,mean_hi,stddev_hi,mean_en,stddev_en])
      signal_feature.append([f1,f2,v])
    elif sys.argv[1]=="All9Signals":
      signal_feature.append([cnt_en,cnt_hin,v,f1,f2,mean_hi,stddev_hi,mean_en,stddev_en])

import random
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
docs = list_sentences_train
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size_word = len(embedding.keys()) + 1
encoded_docs = t.texts_to_sequences(docs)
max_length = 30
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

embedding_matrix = zeros((vocab_size_word, 50))
max_length_char=10
docs_char=[]
for doc in docs:
    #print(doc)
    docs_char.append(" ".join([a.split("_")[0] for a in doc]))

t_char=Tokenizer(char_level=True)
t_char.fit_on_texts(docs_char)

print(t_char.word_index)
vocab_size_char = len(t_char.word_index)+1


encoded_docs_char = []
for doc in docs:
    temp=np.array([])
    for i in range(max_length):
        try:
            word=doc[i]
            temp=np.append(temp,pad_sequences(t_char.texts_to_sequences([word.split("_")[0]]),
                maxlen=10, padding='post')[0])
        except:
            temp=np.append(temp,pad_sequences([[0]*10],maxlen=10, padding='post')[0])
    encoded_docs_char = np.append(encoded_docs_char,np.array(temp))
encoded_docs_char = encoded_docs_char.reshape(len(docs),max_length,max_length_char)

count=0
for word, i in t.word_index.items():
  try:
      embedding_vector = embedding[word]
      embedding_matrix[i] = embedding_vector
  except:
    print(word)
    embedding_matrix[i] = embedding['oov']
    print("OOV here")
    count+=1
print("OOV count:",count)

from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.model_selection import StratifiedKFold
n_folds = 10
skf = StratifiedKFold(n_splits= n_folds ,shuffle=True,random_state=SEED)
acc_sum,f1_sum=0,0
signal_feature=np.array(signal_feature)
labels=np.array(labels)
from keras.callbacks import EarlyStopping
import gc

# np.save('signal_features',signal_feature)
# np.save('y', labels)
# exit()


from keras import optimizers
sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)

#def DataBalance(X):
def returnBalance(X_train_char,X_train_NN,X_train_LSTM,y_train):
    X_train_char_balance,X_train_NN_balance,X_train_LSTM_balance,y_train_balance=[],[],[],[]
    min_cnt=min(sum([i==0 for i in y_train]),sum([i==0 for i in y_train]))
    cnt_0,cnt_1=0,0
    for i in range(len(X_train_char)):
      if y_train[i]==0 and cnt_0<=min_cnt:
        X_train_char_balance.append(X_train_char[i])
        X_train_NN_balance.append(X_train_NN[i])
        X_train_LSTM_balance.append(X_train_LSTM[i])
        y_train_balance.append(y_train[i])
        cnt_0+=1
      if y_train[i]==1 and cnt_1<=min_cnt:
        X_train_char_balance.append(X_train_char[i])
        X_train_NN_balance.append(X_train_NN[i])
        X_train_LSTM_balance.append(X_train_LSTM[i])
        y_train_balance.append(y_train[i])
        cnt_1+=1
    return np.array(X_train_char_balance),np.array(X_train_NN_balance),np.array(X_train_LSTM_balance),np.array(y_train_balance)  


Predicted = np.zeros((518,1))
for i, (train, test) in enumerate(skf.split(padded_docs,labels)):
  #Data preprocessing
  X_train_LSTM = padded_docs[train]
  X_test_LSTM  = padded_docs[test]
  
  X_train_char = encoded_docs_char[train]
  X_test_char = encoded_docs_char[test]

  y_train = labels[train]
  y_test  = labels[test]
  
  X_train_NN=signal_feature[train]
  X_test_NN=signal_feature[test]
  
  #1:10 data balancing perform.to be removed while running for humour
  #deprecated
  #X_train_char,X_train_NN,X_train_LSTM,y_train=returnBalance(X_train_char,X_train_NN,X_train_LSTM,y_train)

  # X_train_LSTM = list(X_train_LSTM)
  # X_test_LSTM = list(X_test_LSTM)
  # for i in range(len(X_train_LSTM)):
  #   X_train_LSTM[i]=[X_train_LSTM[i]]
  # for i in range(len(X_test_LSTM)):
  #         X_test_LSTM[i]=[X_test_LSTM[i]]
  # X_test_LSTM=np.array(X_test_LSTM)
  # X_train_LSTM=np.array(X_train_LSTM)

  X_train_NN=np.array(X_train_NN)
  X_test_NN=np.array(X_test_NN)

    #Shape for the late-incorporation signals
  if sys.argv[1]=="EnHi":
    inputB = Input(shape=(2,))
  elif sys.argv[1]=="Combined":
    inputB = Input(shape=(1,))
  elif sys.argv[1]=="Fraction":
    inputB = Input(shape=(3,))
  elif sys.argv[1]=="All9Signals":
    inputB = Input(shape=(9,))

  word_in = Input(shape=(max_length,))
  emb_word = Embedding(vocab_size_word, 50, weights=[embedding_matrix],
                input_length=max_length, trainable=False)(word_in)
  
  char_in = Input(shape=(max_length, max_length_char,))
  emb_char = TimeDistributed(Embedding(input_dim=vocab_size_char, output_dim=10,
                           input_length=max_length_char))(char_in)
  
  char_enc = TimeDistributed(GRU(units=10, return_sequences=False,recurrent_dropout=0))(emb_char)
  x = concatenate([emb_word, char_enc])
  main_lstm = Bidirectional(GRU(units=50, return_sequences=True,recurrent_dropout=0))(x)
  #Adding attention
  l_att_sent = AttLayer(int(sys.argv[3]))(main_lstm)
  #main_lstm=Flatten()(main_lstm)
  #preds = Dense(1, activation='sigmoid')(l_att)

  #sentEncoder = Model([word_in,char_in], l_att)
  #End here
  #l_att = AttLayer(int(sys.argv[3]))(main_lstm)

  #sentEncoder = Model([word_in,char_in], l_att)

    #Only the part of the sentence encoder changes
  #review_input = Input(shape=(1,max_length), dtype='int32')
  #review_encoder = TimeDistributed(sentEncoder)(review_input)

  # if sys.argv[2]=="GRU":
  #   l_lstm_sent = Bidirectional(GRU(int(sys.argv[3]), return_sequences=True))(l_att)
  # else:
  #   l_lstm_sent = Bidirectional(LSTM(int(sys.argv[3]), return_sequences=True))(l_att)
  # l_att_sent = AttLayer(int(sys.argv[3]))(l_lstm_sent)

  if sys.argv[5]!="ReduceOne":
    preds = Dense(int(sys.argv[4]), activation='relu')(l_att_sent)
  elif sys.argv[5]=="MakeEqual":
    if sys.argv[1]=="EnHi":
      preds = Dense(int(sys.argv[4])-2, activation='relu')(l_att_sent)
    elif sys.argv[1]=="Combined":
      preds = Dense(int(sys.argv[4])-1, activation='relu')(l_att_sent)
    elif sys.argv[1]=="Fraction":
      preds = Dense(int(sys.argv[4])-3, activation='relu')(l_att_sent)
    elif sys.argv[1]=="All9Signals":
      preds = Dense(int(sys.argv[4])-9, activation='relu')(l_att_sent)
  else:
    preds = Dense(int(sys.argv[4]), activation='relu')(l_att_sent)

  model1 = Model([word_in,char_in], preds)

  # if sys.argv[1]=="EnHi":
  #   y = Dense(2, activation="relu")(inputB)
  # elif sys.argv[1]=="Combined":
  #   y = Dense(1, activation="relu")(inputB)
  # elif sys.argv[1]=="Fraction":
  #   y = Dense(3, activation="relu")(inputB)
  # elif sys.argv[1]=="All9Signals":
  #   y = Dense(9, activation="relu")(inputB)
  
  if sys.argv[1]=="EnHi":
    if sys.argv[5]=="ReduceOne":    
      y = Dense(2, activation="relu")(inputB)
      y = Dense(1, activation="sigmoid")(y)
    elif sys.argv[5]=="MakeEqual":
      y = Dense(2, activation="relu")(inputB)
    else:
      y = Dense(2, activation="relu")(inputB)
  elif sys.argv[1]=="Combined":
    # y = Dense(1, activation="relu")(inputB)
    if sys.argv[5]=="ReduceOne":    
      y = Dense(1, activation="relu")(inputB)
      y = Dense(1, activation="sigmoid")(y)
    elif sys.argv[5]=="MakeEqual":
      y = Dense(1, activation="relu")(inputB)
    else:
      y = Dense(1, activation="relu")(inputB)
  elif sys.argv[1]=="Fraction":
    # y = Dense(3, activation="relu")(inputB)
    if sys.argv[5]=="ReduceOne":    
      y = Dense(3, activation="relu")(inputB)
      y = Dense(1, activation="sigmoid")(y)
    elif sys.argv[5]=="MakeEqual":
      y = Dense(3, activation="relu")(inputB)
    else:
      y = Dense(3, activation="relu")(inputB)
  elif sys.argv[1]=="All9Signals":
    if sys.argv[5]=="ReduceOne":    
      y = Dense(9, activation="relu")(inputB)
      y = Dense(1, activation="sigmoid")(y)
    elif sys.argv[5]=="MakeEqual":
      y = Dense(9, activation="relu")(inputB)
    else:
      y = Dense(9, activation="relu")(inputB)

  y3 = Model(inputs=inputB, outputs=y)

  combined = concatenate([model1.output, y3.output])
  z = Dense(1, activation="sigmoid")(combined)
  
  #print(model1.input)
  model = Model(inputs=[model1.input[0],model1.input[1], y3.input], outputs=z)

  #filepath="weights.best."+str(sys.argv[7])+".hdf5"
  filepath="weights_best.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_f1_m', verbose=1, save_best_only=True, mode='max')
  #callbacks_list = [checkpoint]
  earlystop = EarlyStopping(monitor = 'val_f1_m',
                          min_delta = 0,
                          patience = 15,
                          verbose = 1,
                          #restore_best_weights = True
                          )
  
  #callbacks_list = [checkpoint, earlystop]
  callbacks_list = [earlystop]

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
  print(model.summary())

  model.fit([X_train_LSTM,X_train_char, X_train_NN],y_train,epochs=200,validation_split=0.2,callbacks=callbacks_list,batch_size=32)
  #model.load_weights(filepath)

  loss, accuracy, f1_score, precision, recall = model.evaluate([X_test_LSTM,X_test_char, X_test_NN], y_test, verbose=0)
  print("Loss:%f accuracy:%f f1_score:%f  precision:%f recall:%f"%(loss,accuracy,f1_score,precision,recall))
  y_pred = model.predict([X_test_LSTM,X_test_char, X_test_NN])
  
  Predicted = list(zip([yy[0] for yy in list(y_pred)],list(y_test)))
  print("UPDATE ------------------------------------------------")
 
  acc_sum+=accuracy
  f1_sum+=f1_score
  print('Accuracy: %f' % (accuracy*100))
  gc.collect()

  for layer in model.layers:
    print(layer, layer.get_weights())

  break  

ratio = (float(len(labels)-sum(labels)))/sum(labels)

with open(sys.argv[6],"a") as F:
  F.write(sys.argv[7]+"\t"+str(ratio)+"\t"+str(acc_sum/(n_folds*1.00))+"\t"+str(f1_sum/(n_folds*1.00))+"\t"+sys.argv[1]+"\t"+sys.argv[3]+"\t"+sys.argv[4]+"\t"+sys.argv[5]+"\n") 

print("Mean Results:",acc_sum/(n_folds*1.00),"F1_mean:",f1_sum/(n_folds*1.00))

#10e-3>=
