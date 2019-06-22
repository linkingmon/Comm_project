from pygsp import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Matrix operators
#     Graph.W
#     Graph.L
#     Graph.U
#     Graph.D
# Checks
#     Graph.check_weights()                      checks the charateristic of the weights matrix
#     Graph.is_connected
#     Graph.is_directed
# Attributes computation
#     Graph.compute_laplacian([lap_type])	     Compute a graph Laplacian.
#     Graph.estimate_lmax([recompute])	         Estimate the Laplacian’s largest eigenvalue (cached).
#     Graph.compute_fourier_basis([recompute])	 Compute the Fourier basis of the graph (cached).
#     Graph.compute_differential_operator()	     Compute the graph differential operator (cached).
# Differential operators
#     Graph.grad(s)	                             Compute the gradient of a graph signal.
#     Graph.div(s)                               Compute the divergence of a graph signal.
#     Localization
#     Graph.modulate(f, k)	                     Modulate the signal f to the frequency k.
#     Graph.translate(f, i)	                     Translate the signal f to the node i.
# Transforms (frequency and vertex-frequency)
#     Graph.gft(s)	                             Compute the graph Fourier transform.
#     Graph.igft(s_hat)	                         Compute the inverse graph Fourier transform.
#     Graph.gft_windowed(g, f[, lowmemory])	     Windowed graph Fourier transform.
#     Graph.gft_windowed_gabor(s, k)	         Gabor windowed graph Fourier transform.
#     Graph.gft_windowed_normalized(g, f[, lowmemory])	 Normalized windowed graph Fourier transform.
# Plotting
#     Graph.plot(**kwargs)	Plot the graph.
#     Graph.plot_signal(signal, **kwargs)	     Plot a signal on that graph.
#     Graph.plot_spectrogram(**kwargs)	         Plot the graph’s spectrogram.
# Others
#     Graph.get_edge_list()	                     Return an edge list, an alternative representation of the graph.
#     Graph.set_coordinates([kind])              Set node’s coordinates (their position when plotting).
#     Graph.subgraph(ind)                        Create a subgraph given indices.
#     Graph.extract_components()	             Split the graph into connected components.
# Graph models
#     Airfoil(**kwargs)	                          Airfoil graph.
#     BarabasiAlbert([N, m0, m, seed])	          Barabasi-Albert preferential attachment.
#     Comet([N, k])	                              Comet graph.
#     Community([N, Nc, min_comm, min_deg, …])	  Community graph.
#     DavidSensorNet([N, seed])	                  Sensor network.
#     ErdosRenyi([N, p, directed, self_loops, …]) Erdos Renyi graph.
#     FullConnected([N])	                      Fully connected graph.
#     Grid2d([N1, N2])	                          2-dimensional grid graph.
#     Logo(**kwargs)	                          GSP logo.
#     LowStretchTree([k])	                      Low stretch tree.
#     Minnesota([connect])	                      Minnesota road network (from MatlabBGL).
#     Path([N])	                                  Path graph.
#     RandomRegular([N, k, maxIter, seed])	      Random k-regular graph.
#     RandomRing([N, seed])	                      Ring graph with randomly sampled nodes.
#     Ring([N, k])	                              K-regular ring graph.
#     Sensor([N, Nc, regular, n_try, distribute, …])	  Random sensor graph.
#     StochasticBlockModel([N, k, z, M, p, q, …])	      Stochastic Block Model (SBM).
#     SwissRoll([N, a, b, dim, thresh, s, noise, …])	  Sampled Swiss roll manifold.
#     Torus([Nv, Mv])	                                  Sampled torus manifold.
# Nearest-neighbors graphs constructed from point clouds
#     NNGraph(Xin[, NNtype, use_flann, center, …])	     Nearest-neighbor graph from given point cloud.
#     Bunny(**kwargs)	                                 Stanford bunny (NN-graph).
#     Cube([radius, nb_pts, nb_dim, sampling, seed])	 Hyper-cube (NN-graph).
#     ImgPatches(img[, patch_shape])	                 NN-graph between patches of an image.
#     Grid2dImgPatches(img[, aggregate])	             Union of a patch graph with a 2D grid graph.
#     Sphere([radius, nb_pts, nb_dim, sampling, seed])	 Spherical-shaped graph (NN-graph).
#     TwoMoons([moontype, dim, sigmag, N, sigmad, …])	 Two Moons (NN-graph).


class spectral_filter():
    def __init__(self, scale, upbound):
        self.scale = scale
        self.upbound = upbound

    def h(self, lamda):
        return lamda <= self.upbound*self.scale

    def evaluate(self, e):
        return self.h(e)

    def plot_filter(self):
        ax = np.arange(0, 2.01, 0.01)
        plt.plot(ax, self.h(ax))
        plt.show()


class clustering():

    def __init__(self, G):
        self.G = G
        self.G.compute_fourier_basis()

    # directly apply SVD on Laplacian matrix
    def pca_reduction(self, dim):
        self.pca = PCA(n_components=dim, copy=False, svd_solver='full', random_state=100).fit_transform(self.G.L.T.todense())
        print("Shape of pca reduction: ", self.pca.shape)

    # calculate fourier feature vector
    def fourier_feature_vector(self, filters):
        self.filters = filters
        self.h = self.filters.evaluate(self.G.e)
        self.d = np.diag(self.h)
        self.feature = np.dot(np.dot(self.G.U, self.d), self.G.U.T)
        print("Shape of fourier feature vector: ", self.feature.shape)
        return self.feature

    # apply K-means for clustering and plot the result
    def plot_clust(self, classes):
        labels_pca = KMeans(n_clusters=classes, random_state=100).fit(self.pca).labels_
        labels_gft = KMeans(n_clusters=classes, random_state=100).fit(self.feature.T).labels_
        fig, axes = plt.subplots(1, 2)
        self.G.plot_signal(labels_pca, ax=axes[0], show_edges=True)
        self.G.plot_signal(labels_gft, ax=axes[1], show_edges=True)
        plt.show()

    # plot the filter that is used in the fourier feature vector method
    def plot_filter(self):
        y = self.filters.evaluate(self.G.e)
        plt.plot(self.G.e, y)
        plt.show()


if __name__ == '__main__':
    G = graphs.Community(N=150, Nc=5, comm_sizes=[40, 5, 30, 35, 40], seed=42, lap_type='normalized', world_density=0.1, comm_density=0.8)
    C = clustering(G)
    C.pca_reduction(dim=50)
    f = spectral_filter(scale=0.4, upbound=2)
    C.fourier_feature_vector(filters=f)
    C.plot_clust(classes=5)
