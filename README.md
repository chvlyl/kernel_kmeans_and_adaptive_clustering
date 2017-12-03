# Kernel k-means and adaptive non-parametric clustering

- data/*.txt: 8 simulated data downloaded from https://cs.joensuu.fi/sipu/datasets/ under shape data section. Another 8 real data downloaded from the UCI website (http://archive.ics.uci.edu/ml/). Features of real data were scaled to [0,1] if they are in different scale.

- kmean_clustering/kmean_clustering.py: K means, kernel K means and adaptive clustering were implemented and wrapped into a python package. The Euclidean distance was used in three clustering algorithm. For kernel K means, the gaussian kernel was used.

- notebook/real_data_analysis.ipynb:

- report/report.pdf:



In order to compare the three implemented clustering methods, eight simulated data were downloaded from https://cs.joensuu.fi/sipu/datasets/ under shape data section. Another eight real data were download from the UCI website (http://archive.ics.uci.edu/ml/). Features of real data were normalized to [0,1] if they are in different scale. The Euclidean distance measure was used in three clustering algorithm. For kernel K means, the gaussian kernel was used. The parameter $\sigma$ was tuned by searching in ($1.0,2.0,3.0,5.0$) and the best result was reported.   In order to find the best cluster numbers for  K means and kernel K means clustering, values from $[max(2,true\_n\_clusters-5),true\_n\_clusters+5]$ were tested and the Silhouette scores were calculate. The best cluster number was selected with the largest average Silhouette score. Since the true cluster labels are available, the Rand index was used to compare the performance of the three clustering algorithms. The Rand index compares data points in the predicted and true clusters and computes a similarity measure by counting data points assigned into the same and different clusters. Check the notebook for details about data analysis. 
