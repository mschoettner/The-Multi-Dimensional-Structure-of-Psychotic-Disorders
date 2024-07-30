import pandas as pd
import numpy as np
import multiprocessing as mp

from sklearn.cluster import KMeans, SpectralClustering

def consensus_clustering(data, method, iterations, frac, clusters, verbose=False):
    """Consensus clustering
    data: data matrix as pandas dataframe
    method: clustering algorithm to use
    frac: fraction of data to resample
    iterations: how many times the data should be resampled
    clusters: range of how many clusters, tuple(minimum, maximum)
    verbose: verbosity, bool
    """
    # object to store consensus matrices
    Mks = []
    # iterate over possible numbers of clusters
    for k in range(clusters[0],clusters[1]+1):
        if verbose == True:
            print("Number of clusters:", k)
        M = pd.DataFrame(np.zeros((data.shape[0], data.shape[0])),
                              index=data.index, columns=data.index)
        I = pd.DataFrame(np.zeros((data.shape[0], data.shape[0])),
                              index=data.index, columns=data.index)
        for h in range(iterations):
            if verbose == True:
                print("Iteration", h+1)
            # draw sample
            sample = data.sample(frac=frac)
            # save indices
            indices_sampled = list(sample.index)
            # run clustering
            if method == KMeans:
                labels = method(n_clusters=k, n_init="auto").fit_predict(sample)
            else:
                labels = method(n_clusters=k).fit_predict(sample)
            # combine indices and labels to Series
            prediction = pd.Series(labels, indices_sampled)
            # compute connectivity matrix and indicator matrix
            Mh = pd.DataFrame(np.zeros((data.shape[0], data.shape[0])),
                              index=data.index, columns=data.index)
            Ih = pd.DataFrame(np.zeros((data.shape[0], data.shape[0])),
                              index=data.index, columns=data.index)
            for i in data.index: # loop over all indices
                for j in data.index: # again
                    if i not in indices_sampled or j not in indices_sampled:
                        Mh.loc[i,j] = 0.0
                        Ih.loc[i,j] = 0.0
                    elif prediction.loc[i] == prediction.loc[j]:
                        Mh.loc[i,j] = 1.0
                        Ih.loc[i,j] = 1.0
                    elif prediction.loc[i] != prediction.loc[j]:
                        "now!"
                        Mh.loc[i,j] = 0.0
                        Ih.loc[i,j] = 1.0
            # update consensus and indicator matrices
            M += Mh
            I += Ih
        Mk = M/I
        # when two variables never occur in the same cluster, we have 0/0=NaN, assert to 0
        Mk.fillna(0.0, inplace=True)
        Mks.append(Mk)
    return(Mks)

def cluster(data, k, method, iterations, frac, verbose):
    # if verbose == True:
    #     print("Number of clusters:", k, flush=True)
    M = pd.DataFrame(np.zeros((data.shape[0], data.shape[0])),
                            index=data.index, columns=data.index)
    I = pd.DataFrame(np.zeros((data.shape[0], data.shape[0])),
                            index=data.index, columns=data.index)
    for h in range(iterations):
        # if verbose == True:
        #     print("Iteration", h+1, flush=True)
        # draw sample
        sample = data.sample(frac=frac)
        # save indices
        indices_sampled = list(sample.index)
        # run clustering
        if method == KMeans:
            labels = method(n_clusters=k, n_init="auto").fit_predict(sample)
        else:
            labels = method(n_clusters=k).fit_predict(sample)
        # combine indices and labels to Series
        prediction = pd.Series(labels, indices_sampled)
        # compute connectivity matrix and indicator matrix
        Mh = pd.DataFrame(np.zeros((data.shape[0], data.shape[0])),
                            index=data.index, columns=data.index)
        Ih = pd.DataFrame(np.zeros((data.shape[0], data.shape[0])),
                            index=data.index, columns=data.index)
        for i in data.index: # loop over all indices
            for j in data.index: # again
                if i not in indices_sampled or j not in indices_sampled:
                    Mh.loc[i,j] = 0.0
                    Ih.loc[i,j] = 0.0
                elif prediction.loc[i] == prediction.loc[j]:
                    Mh.loc[i,j] = 1.0
                    Ih.loc[i,j] = 1.0
                elif prediction.loc[i] != prediction.loc[j]:
                    "now!"
                    Mh.loc[i,j] = 0.0
                    Ih.loc[i,j] = 1.0
        # update consensus and indicator matrices
        M += Mh
        I += Ih
    Mk = M/I
    # when two variables never occur in the same cluster, we have 0/0=NaN, set to 0
    Mk.fillna(0.0, inplace=True)
    return Mk
        

def consensus_clustering_parallel(data, method=KMeans, iterations=10, frac=0.8, clusters=(2,10), verbose=False):
    """Consensus clustering
    data: data matrix as pandas dataframe
    method: clustering algorithm to use
    frac: fraction of data to resample
    iterations: how many times the data should be resampled
    clusters: range of how many clusters, tuple(minimum, maximum)
    verbose: verbosity, bool
    """
    # iterate over possible numbers of clusters
    pool = mp.Pool(mp.cpu_count())
    Mks = [pool.apply(cluster, args=(data, k, method, iterations, frac, verbose)) for k in range (clusters[0],clusters[1]+1)]
    pool.close()
    return(Mks)

def consensus_score(matrix, k):
    """
    Computes the consensus score of a consensus matrix.
    
    Parameters
    ----------
    matrix: np.array
        The consensus matrix that is to be scored.
    k: int or float
        Number of clusters the consensus matrix is based on.
    
    Returns
    -------
    score: float
        The consensus score.
    """
    # flatten consensus matrix
    vector = matrix.flatten()
    # do mean split
    #mean = np.mean(vector)
    cutoff = 1/k
    # get upper and lower values
    upper = []
    lower = []
    for value in vector:
        if value > cutoff:
            upper.append(value)
        elif value < cutoff:
            lower.append(value)
    # average
    upper_mean = np.mean(upper)
    lower_mean = np.mean(lower)
    # compute difference
    score = upper_mean - lower_mean
    
    return score

def calculate_cluster_consensus(Mks, method):
    """
    Clustering of the consensus matrices.
    :param Mks: Consensus matrices as list of dataframes
    :return: Cluster consensus scores as dataframe and predictions as list of dictionaries
    """
    labels = []
    for i, m in enumerate(Mks):
        labels.append(method(n_clusters=i + 2).fit_predict(1 - m))

    labels = np.array(labels)

    cluster_consensus = []
    predictions = []
    counter = 0
    for i, l in enumerate(labels):
        l_df = pd.DataFrame(l, columns=["cluster"], index=Mks[i].columns)
        n_clusters = len(np.unique(l))
        print("Number of clusters:", n_clusters, "\n")
        pred_dict = {}
        for c in range(n_clusters):
            variables_in_cluster = l_df.index[l_df["cluster"] == c]
            # print("Variables in cluster", c, ":\n", variables_in_cluster)
            pred_dict[f'cluster_{c}'] = variables_in_cluster
            consensus_values = np.array(Mks[i].loc[variables_in_cluster, variables_in_cluster])
            upper_indices = np.triu_indices(consensus_values.shape[0], 1)
            cluster_cons = np.mean(consensus_values[upper_indices])
            cluster_consensus.append([n_clusters, c, cluster_cons])
            # print("Cluster consensus:", cluster_cons, "\n")
            counter += 1
        predictions.append(pred_dict)

    cluster_consensus = pd.DataFrame(cluster_consensus, columns=["n_clusters", "cluster", "consensus"])

    return cluster_consensus, predictions, labels