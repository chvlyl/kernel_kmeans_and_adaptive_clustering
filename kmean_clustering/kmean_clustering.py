import numpy as np
import random,sys
from  scipy.spatial.distance import pdist,squareform

def test():
	print(1)
    
    
def polynomial_kernel(data,p=2,gamma=1.0):
    return (np.dot(data,data.T)+gamma)**p


def gaussian_kernel(data,sigma=1.0):
    ### 'sqeuclidean': squared Euclidean distance
    return np.exp(-0.5/(sigma**2)*squareform(pdist(data,'sqeuclidean')))

def k_means(data, n_clusters=3, n_init=10, max_iter=100, verbose=False):
    '''
    data: a numeric numpy array
    n_clusters: number of clusters
    n_init: number of different initializations to run kmeans
    max_iter: number of max iterations 
    verbose: output detailed information
    '''
    ### may not be efficient in terms of memory use
    ### no need to save whole history
    ### get whole hitory for debugging purpose
    controid_history = {}
    cluster_label_history = {}
    sse_history = np.zeros(shape=(n_init,1))
    ### start k-means
    n_points = data.shape[0]
    ### repeat k-means n_init times 
    ### return the best one 
    for i_init in range(n_init):
        if verbose: print('Random seed',i_init)
        #### set random seed
        np.random.seed(i_init)
        #### generate initial cluster labels
        cluster_labels = np.random.choice(range(n_clusters),size=n_points, replace=True)
        #### generate initial centroids
        #### randomly choose n_clusters points from the data as centroids
        centroids = data[np.random.choice(np.arange(n_points), n_clusters, replace=False),:]
        for i_iter in range(max_iter):
            if verbose: print('Iteration',i_iter,end=', ')
            distance_to_centroids = np.zeros(shape=(data.shape[0],n_clusters))
            for i_centroid in range(n_clusters):
                ### ord=2 is L2 distance
                ### axis=1 is to calculate norm along columns
                distance_to_centroids[:,i_centroid] = np.linalg.norm(data-centroids[i_centroid,:],ord=2,axis=1)
                #break
            ### assign the cluster labels
            cluster_labels = np.argmin(distance_to_centroids,axis=1)
            sse = np.sum((np.min(distance_to_centroids,axis=1))**2)
            if verbose: print('SSE',sse)
            ### re-calculate centroids
            previous_centroids = centroids
            centroids = np.array([data[cluster_labels == i_centroid].mean(axis = 0) for i_centroid in range(n_clusters)])
            ### if centroids don't change
            ### stop the iteration
            if np.all(previous_centroids == centroids):
                if verbose: print('Centroids do not change',i_iter)
                break
        controid_history[i_init] = centroids
        cluster_label_history[i_init] = cluster_labels
        sse_history[i_init] = sse
    ### find the best initializations
    best_iter = np.argmin(sse_history)
    best_sse = sse_history[best_iter]
    best_controids = controid_history[best_iter]
    best_cluster_label = cluster_label_history[best_iter]
    
    return {'best_iter':best_iter,
            'best_sse':best_sse,
            'best_controids':best_controids,
            'best_cluster_label':best_cluster_label,
            'controid_history':controid_history,
            'cluster_label_history':cluster_label_history,
            'sse_history':sse_history,
           }



def k_means_kernel(data, n_clusters=3, n_init=20, max_iter=100, kernel=None,verbose=False):
    '''
    data: a numeric numpy array
    n_clusters: number of clusters
    n_init: number of different initializations to run kmeans
    max_iter: number of max iterations 
    verbose: output detailed information
    kernel: "None", regular k means; "gaussian",  k means with gaussian kernel
    '''
    ### may not be efficient in terms of memory use
    ### no need to save whole history
    ### get whole hitory for debugging purpose
    controid_history = {}
    cluster_label_history = {}
    sse_history = np.zeros(shape=(n_init,1))
    ### start k-means
    n_points = data.shape[0]
    if kernel == 'gaussian':
        ### calculate the kernel matrix
        kernel_matrix = gaussian_kernel(data)
    ### repeat k-means n_init times 
    ### return the best one 
    for i_init in range(n_init):
        if verbose: print('Random seed',i_init)
        #### set random seed
        np.random.seed(i_init)
        #### generate initial cluster labels
        cluster_labels = np.random.choice(range(n_clusters),size=n_points, replace=True)
        #### generate initial centroids
        #### randomly choose n_clusters points from the data as centroids
        centroids = data[np.random.choice(np.arange(n_points), n_clusters, replace=False),:]
        for i_iter in range(max_iter):
            if verbose: print('Iteration',i_iter,end=', ')
            distance_to_centroids = np.zeros(shape=(data.shape[0],n_clusters))
            for i_centroid in range(n_clusters):
                if kernel is None:
                    ### ord=2 is L2 distance
                    ### axis=1 is to calculate norm along columns
                    distance_to_centroids[:,i_centroid] = np.linalg.norm(data-centroids[i_centroid,:],ord=2,axis=1)
                elif kernel == 'gaussian':
                    kth_cluster_ind = (cluster_labels == i_centroid).astype('int')
                    kth_cluster_matrix = np.outer(kth_cluster_ind,kth_cluster_ind)
                    dist1 = np.diag(kernel_matrix)
                    dist2 = 2.0*np.sum(np.tile(kth_cluster_ind,(n_points,1))*kernel_matrix,axis=1)/np.sum(kth_cluster_ind)
                    dist3 = np.sum(kth_cluster_matrix*kernel_matrix)/np.sum(kth_cluster_matrix)
                    #print(dist1.shape,dist2.shape,dist3.shape,)
                    ### ord=2 is L2 distance
                    ### axis=1 is to calculate norm along columns
                    distance_to_centroids[:,i_centroid] = dist1-dist2+dist3
                    #break
                else:
                    sys.exit('Kernel parameter is not correct!')
            #print(distance_to_centroids)
            ### assign the cluster labels
            cluster_labels = np.argmin(distance_to_centroids,axis=1)
            sse = np.sum((np.min(distance_to_centroids,axis=1))**2)
            if verbose: print('SSE',sse)
            ### re-calculate centroids
            previous_centroids = centroids
            centroids = np.array([data[cluster_labels == i_centroid].mean(axis = 0) for i_centroid in range(n_clusters)])
            ### if centroids don't change
            ### stop the iteration
            if np.all(previous_centroids == centroids):
                if verbose: print('Centroids do not change',i_iter)
                break
            #break
        controid_history[i_init] = centroids
        cluster_label_history[i_init] = cluster_labels
        sse_history[i_init] = sse
        #break
    ### find the best initializations
    best_iter = np.argmin(sse_history)
    best_sse = sse_history[best_iter]
    best_controids = controid_history[best_iter]
    best_cluster_label = cluster_label_history[best_iter]
    
    return {'best_iter':best_iter,
            'best_sse':best_sse,
            'best_controids':best_controids,
            'best_cluster_label':best_cluster_label,
            'controid_history':controid_history,
            'cluster_label_history':cluster_label_history,
            'sse_history':sse_history,
           }