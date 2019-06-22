# Comm_project

## Spectral clustering on Community graph
#### related files
1. community.py: compare pca and spectral method on laplacian matrix of an unweighted community graph
#### results
<img src = 'https://github.com/linkingmon/Comm_project/blob/master/figure/Figure_1.png'><br>
<img src = 'https://github.com/linkingmon/Comm_project/blob/master/figure/Figure_2.png' width=170%><br><br>

## Comparison between PCA+Kmeans and Spectral clustering
#### related files
1. download models and data from: https://drive.google.com/open?id=1BtpzlMUzz1y461WzCk2GvUyPDHpa2Gsv
2. autoencdoer.py: the autoencoder model build by keras
3. autoencoder_predict.py: read the image data and run the pretrained-autoencoder model to output the prediction
4. pca_clustering.py: apply pca reduction method through the output of autoencoder, then apply Kmeans to serparate into 2 clusters
5. spectral clustering.py: apply cosine-similarity as edge weight and run spectral clustering by its eigenvectors
6. test.py: calculate the correct rate of the two method.

#### results
Since it takes a long time to run through the whole data, we apply Kmeans only on 1000 / 3000 / 4000 / 5000 / 7000 / 10000 datas seperately.<br>
PCA correct rate: 0.505 / 0.695 / 0.535 / 0.7 / 0.7 / 0.7<br>
GFT correct rate: 0.65 / 0.655 / 0.645 / 0.659 / 0.659 / 0.699<br>
We can see that PCA-method has better accuracy, but has lower stability compared with GFT-method.<br><br>

## Comparison between Kernighan-Lin algorithm and spectral clustering on Community graph
KL algorithm is widely used Heuristic in phsical design partitioning which helps to find the min-cost cut in a graph.<br>
Clustering Community graph is also aimed to fidn the min-cut in the graph.<br>
