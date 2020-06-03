import numpy as np
import scipy.stats

X = np.load('signal_features.npy')
Y = np.load('y.npy')

for i in range(9):
	print(scipy.stats.pearsonr(X[:,i], Y)[0], end=',')    # Pearson's r
	#print(scipy.stats.spearmanr(X[:,i], Y))   # Spearman's rho
	#print(scipy.stats.kendalltau(X[:,i], Y))