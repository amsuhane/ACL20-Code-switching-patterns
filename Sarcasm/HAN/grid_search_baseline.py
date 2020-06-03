import os
import gc
import random
import sys

random.seed(42)


mode=["EnHi"] #just a dummy, not actually used as baseline
hidden_sizes=[i for i in range(20,100,10)]
Late_Incorp_Modes=["ReduceOne","MakeEqual"]
combined_layer_size=[15,20,25,30]
data_files = ["../Datasets/dataset_sarcasm_processed_6.pkl", "../datasets/dataset_sarcasm_processed_7.pkl"]

model=["GRU"]
Op_file = "./grid_search_baseline.csv"


with open(Op_file,"w") as F:
    F.write("Dataset\tRatio\tAcc.\tF1\tmode\thidden_size\tlayer_size\tlate_mode\n")

for j in hidden_sizes:
    for k in model:
        for i in mode:
            for l in combined_layer_size:
                for late_mode in Late_Incorp_Modes:
                    for data_file in data_files:
                        os.system("python2 training_sarcasm_HAN_baseline_char_level_1.py "+str(i)+" "+str(k)+" "
                                    +str(j)+" "+str(l)+" "+str(late_mode)+" "+str(Op_file)+" "+str(data_file))
                        print("python2 training_humour_HAN_baseline_char_level_1_.py "+str(i)+" "+str(k)+" "
                                    +str(j)+" "+str(l)+" "+str(late_mode)+" "+str(Op_file)+" "+str(data_file))                    
gc.collect()