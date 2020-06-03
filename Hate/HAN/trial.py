import pickle
import numpy as np
import sys

with open('../Dataset/embedding_final_all.pkl','rb') as F:
	embedding=pickle.load(F)#, encoding="bytes")

with open("../Dataset/dataset_hate_processed_manual_annotated.pkl",'rb') as F:
	data1=pickle.load(F)

list_sentences_train=[]
y=[]
labels=[]
signal_feature=[]
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

print(signal_feature[0])