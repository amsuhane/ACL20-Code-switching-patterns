import os
import gc
import random
import sys

#random.seed(42)

feature_lens = [20, 30, 50, 100, 150, 200, 350, 500, 750, 1000, 1200]
random_states = [42]

with open('hate.csv',"w") as F:
    F.write("Baseline/Switch\tLeakage\tfeature_len\trandom_state\tAcc\tF1\n")

for feature_len in feature_lens:
    for random_state in random_states:
        os.system("python recreate_results.py "+str(feature_len)+" "+str(random_state))
        print("python recreate_results.py "+str(feature_len)+" "+str(random_state))
gc.collect()
