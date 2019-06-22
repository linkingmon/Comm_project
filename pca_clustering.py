from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA


dim = 2
kmeans_num = 7000

X = np.load('predict.npy')
print("Shape of X: ", X.shape)
pca = PCA(n_components=100, copy=False, whiten=True, svd_solver='full', random_state=100).fit_transform(X[:kmeans_num])
print("Shape after PCA reduction: ", pca.shape)

k = KMeans(n_clusters=2, random_state=100).fit(pca)
print('ksum:', k.labels_.sum())
np.save('Label.npy', k.labels_)
