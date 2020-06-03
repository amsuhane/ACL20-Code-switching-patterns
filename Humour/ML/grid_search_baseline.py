import os
import gc
import random
import sys

"""
Parameters used for grid search
vector_sizes = [20, 30, 50, 100, 150, 200, 250, 400, 500]
random_states = [42, 1973, 53022]
"""

# Best parameters choosen after grid search
vector_sizes = [150]
random_states = [42]

with open("results_baseline.csv", 'w') as f:
	f.write("Type\trandom_state\tvector_length\taccuracy\tF1\tmicro_f1\tmacro_f1\n")

for vector_size in vector_sizes:
	for random_state in random_states:
		print(vector_size, random_state)
		os.system("python recreate_results.py " + "All9Signals " + str(vector_size) + " " + str(random_state))