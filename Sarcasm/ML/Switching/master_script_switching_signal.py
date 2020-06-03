import os
import gc
import random
import sys

random.seed(42)

#mode=["EnHi","Combined","Fraction","All9Signals"]
mode=["All9Signals"]

hidden_sizes=[50]#[i for i in range(20,100,10)]
Late_Incorp_Modes=["ReduceOne"] #["ReduceOne","MakeEqual"]
combined_layer_size=[15]#[15,20,25,30]
model=["GRU"]
try:
    os.mkdir("Final_Compilations")
except:
    print("Directory already present")
Op_file = "./Final_Compilations/"+sys.argv[2]

data_file = sys.argv[1]#'dataset_hate_processed.pkl'#'dataset_humour_processed.pkl'#'dataset_sarcasm_processed.pkl'   

with open(Op_file,"w") as F:
    F.write("Mode\tLSTM_or_GRU\tHidden_Size\tPre-final-layer-size\tLate_Incorp_Mode\tAcc.\tF1\n")

for j in hidden_sizes:
    for k in model:
        for i in mode:
            for l in combined_layer_size:
                for late_mode in Late_Incorp_Modes:
                    os.system("python2 training_sarcasm_HAN_switching_signal_char_level_1_.py "+str(i)+" "+str(k)+" "
                                +str(j)+" "+str(l)+" "+str(late_mode)+" "+str(Op_file)+" "+str(data_file))

gc.collect()
