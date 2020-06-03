import os
import gc
import random
import sys

random.seed(42)


"""
Parameters used for grid searched
mode=["EnHi","Combined","Fraction","All9Signals"]
hidden_sizes=[i for i in range(20,100,10)]
Late_Incorp_Modes=["ReduceOne","MakeEqual"]
combined_layer_size=[15,20,25,30]
model=["GRU"]
Op_file = "./Final_Compilations/"+sys.argv[2]
"""

# Best parameters after grid search
mode=["All9Signals"]
hidden_sizes=[30]
Late_Incorp_Modes=["ReduceOne"]
combined_layer_size=[30]
model=["GRU"]
Op_file = "results_HAN_switching.csv"
data_file = "../Dataset/dataset_humour_processed.pkl"

with open(Op_file,"w") as F:
    F.write("Mode\tLSTM_or_GRU\tHidden_Size\tPre-final-layer-size\tLate_Incorp_Mode\tAcc.\tF1\n")

for j in hidden_sizes:
    for k in model:
        for i in mode:
            for l in combined_layer_size:
                for late_mode in Late_Incorp_Modes:
                    os.system("python2 training_humour_HAN_switching_signal_char_level_1_.py "+str(i)+" "+str(k)+" "
                                +str(j)+" "+str(l)+" "+str(late_mode)+" "+str(Op_file)+" "+str(data_file))

gc.collect()
