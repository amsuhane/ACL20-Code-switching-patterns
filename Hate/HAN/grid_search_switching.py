import os
import gc
import random
import sys

random.seed(42)


mode=["All9Signals"] 
hidden_sizes=[20] #[i for i in range(20,100,10)]
Late_Incorp_Modes=["MakeEqual"]
combined_layer_size=[20] #[15,20,25,30]

data_files = ["../Dataset/dataset_hate_processed_manual_annotated.pkl"]
model=["GRU"]
Op_file = "./grid_search_switching.csv"


with open(Op_file,"w") as F:
    F.write("Dataset\tRatio\tAcc.\tF1\tmode\thidden_size\tlayer_size\tlate_mode\n")

for j in hidden_sizes:
    for k in model:
        for i in mode:
            for l in combined_layer_size:
                for late_mode in Late_Incorp_Modes:
                    for data_file in data_files:
                        os.system("python2 training_hate_HAN_switching_signal_char_level_1_.py "+str(i)+" "+str(k)+" "
                                    +str(j)+" "+str(l)+" "+str(late_mode)+" "+str(Op_file)+" "+str(data_file))
                        print("python2 training_hate_HAN_switching_signal_char_level_1_.py "+str(i)+" "+str(k)+" "
                                    +str(j)+" "+str(l)+" "+str(late_mode)+" "+str(Op_file)+" "+str(data_file))                    
gc.collect()
