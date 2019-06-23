import numpy as np
import pandas as pd

pca = np.load('Label.npy')
print("Shape of pca predict: ", pca.shape)
gft = np.load('Label_gft.npy')
print("Shape of gft predict: ", gft.shape)

ans = np.array([0]*2500+[1]*2500)
print("Shape of true answers: ", ans.shape)
print("Percantage of (class 0,class 1) in the test set is : ", "({},{})".format(sum(ans == 1)/len(ans), sum(ans == 0)/len(ans)))

rate_pca = np.sum(pca[:len(ans)] != ans) / len(ans)
rate_gft = np.sum(gft[:len(ans)] != ans) / len(ans)

print("PCA correct rate: ", rate_pca if (rate_pca > 0.5) else (1-rate_pca))
print("GFT correct rate: ", rate_gft if (rate_gft > 0.5) else (1-rate_gft))
