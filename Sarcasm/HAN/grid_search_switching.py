import os
import gc
import random
import sys

random.seed(42)



mode=["All9Signals"]
hidden_sizes=[30]
Late_Incorp_Modes=["ReduceOne"]
combined_layer_size=[25]
data_files = ["../Datasets/dataset_sarcasm_processed_6.pkl"]

model=["GRU"]
Op_file = "./grid_search_switching_new.csv"


with open(Op_file,"w") as F:
    F.write("Dataset\tRatio\tAcc.\tF1\tmode\thidden_size\tlayer_size\tlate_mode\n")

for j in hidden_sizes:
    for k in model:
        for i in mode:
            for l in combined_layer_size:
                for late_mode in Late_Incorp_Modes:
                    for data_file in data_files:
                        os.system("python2 training_sarcasm_HAN_switching_signal_char_level_1_.py "+str(i)+" "+str(k)+" "
                                    +str(j)+" "+str(l)+" "+str(late_mode)+" "+str(Op_file)+" "+str(data_file))
                        print("python2 training_humour_HAN_switching_signal_char_level_1_.py "+str(i)+" "+str(k)+" "
                                    +str(j)+" "+str(l)+" "+str(late_mode)+" "+str(Op_file)+" "+str(data_file))                    
gc.collect()