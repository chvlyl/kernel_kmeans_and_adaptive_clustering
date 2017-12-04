# Kernel k-means and adaptive non-parametric clustering

- data/*.txt: 8 simulated data downloaded from https://cs.joensuu.fi/sipu/datasets/ under shape data section. Another 8 real data downloaded from the UCI website (http://archive.ics.uci.edu/ml/). Features of real data were scaled to [0,1] if they are in different scale.

- kmean_clustering/kmean_clustering.py: K means, kernel K means and adaptive clustering were implemented and wrapped into a python package. The Euclidean distance was used in three clustering algorithms. For kernel K means, the Gaussian kernel was used.

- notebook/real_data_analysis.ipynb: The detailed analysis were saved in the notebook. I recommend to use the [Table of Contents plugin in Jupyter notebook](http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/toc2/README.html) to navigate this notebook.

In order to find the best cluster numbers for  K means and kernel K means clustering, values from [max(2,true\_n\_clusters-5),true\_n\_clusters+5] were tested and the Silhouette scores were calculate. The best cluster number was selected with the largest average Silhouette score. For a given number of clusters, The parameter sigma in Gaussian kernel was tuned by searching in (1.0,2.0,3.0,5.0) and the best result was reported. The parameter lambda in the adpative clustering was tuned by searching in (0.1,0.5,1.0,2.0,5.0) and the best lambda was selected by largest average Silhouette score. The Rand index was used to compare the performance of the three clustering algorithms. 

- report/report.pdf: A brief summary of the analysis.

- notebook/why_adaptive_clustering_does_not_work_on_real_data.ipynb: The adaptive clustering works quite well on the simulated data (note that p=2 for all simulated data), which indicates the implementation of this algorithm should be OK. However, this approach does not work well on the real data (p>5 for most of the data). I noticed two possible reasons: 1. The weight matrix fails to initialize properly for relative large p (most of the weight matrix values are initialized to 1). The paper proposed to set n0=2p+2 and then set h accordingly. This may be too large for relative large p. 2. Even with very small lambda such as 0.0001 (since T follows  the Chi-square distribution under the null, the test statistic should be non-negative and thus negative lambda is not meanful), the weight matrix still grows (some values become 1) in the large p setting. This is either related to bad initialization or the test statistic they use.  The test statistic follows a Chi-square distribution under the null and the Chi-square test needs the degree of freedom in order to calculate the p value. However, no such information was considered in their algorithm. This notebook was aimed to investigate those problems. It seems to me that the default n0=2p+2 proposed in the paper is a bad initialization value for relatively large p. We need set smaller n0 and smaller lambda.


- notebook/proflie_implemented_functions.ipynb: This is to profile each component of the implemented functions to identify the most time consuming part and make the implementaion more efficient.



