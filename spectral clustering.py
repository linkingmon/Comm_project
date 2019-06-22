from sklearn.cluster import KMeans
import numpy as np
import utils

dim = 2
kmeans_num = 7000

X = np.load('predict.npy')
print("Shape of X: ", X.shape)
sim = utils.compute_similarity_matrix(X[:kmeans_num])
(eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(sim)
spectral_embeddings = eigenvectors[:, :dim]
k = KMeans(n_clusters=dim, max_iter=300, random_state=100).fit(spectral_embeddings)
print('ksum:', k.labels_.sum())
np.save('Label_gft.npy', k.labels_)
