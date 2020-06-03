import os
import gc
import random
import sys

vector_sizes = [21, 25, 31, 41, 51, 91, 141, 191, 241, 391, 491]

with open("results.csv", 'w') as f:
	f.write("vector_sizes\tAcc.\tf1_macro\tf1_micro\n")

for i in vector_sizes:
	os.system("python2 classification.py " + "randomforest " + str(i))