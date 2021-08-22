import pandas as pd
from os import path
from sys import argv
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns


def k_Medoids(dataframe, max_iters, n_cluster):

    print(str(n_cluster)+"-Medoids clustering...")
    cluster = KMedoids(n_clusters=n_cluster, metric='euclidean',
                       init='build', method='alternate', max_iter=max_iters).fit(dataframe)

    """cluster = KMeans(n_clusters=n_cluster,init='k-means++', n_init=20,
                     algorithm='elkan', max_iter=max_iters).fit(dataframe)"""

    print(cluster.inertia_,"is the cost of", n_cluster, "clusters with", max_iters, "iterations")
    return cluster


def elbow(dframe, iteration, k):

    # Sum of distances of samples to their closest cluster center.
    distances = {}

    for i in range(2, k+1):
        medoid = k_Medoids(dframe, iteration, i)
        distances[i] = medoid.inertia_

    plt.title('elbow')
    plt.xlabel('k')
    plt.ylabel('distance')
    sns.pointplot(x=list(distances.keys()), y=list(distances.values()))
    plt.show()


def main(data, itero, elbow_k):
    try:
        if not path.isfile(data):
            print("file not found or wrong directory, returning")
            return

    except IndexError:
        print("Error with number of arguments")
        return

    try:
        if itero > 0:
            max_iter = int(itero)
        else:
            print("incorrect iterations value")
            return
    except IndexError:
        max_iter = 10
    try:
        if elbow_k > 0:
            k = int(elbow_k)
        else:
            print("incorrect k value for elbow plotting")
            return
    except IndexError:
        k = 5

    pd.set_option('expand_frame_repr', False)
    dataframe = pd.read_csv(data)

    # elbow(dataframe, max_iter, k)

    cluster = k_Medoids(dataframe, max_iter, k)
    k_mean_labels = cluster.labels_

    bay_frame = pd.read_csv('../datasets/bayesian_ready.csv').assign(n_cluster=k_mean_labels)
    prolog_frame = pd.read_csv('../datasets/prolog_ready.csv').assign(n_cluster=k_mean_labels)

    bay_frame.to_csv('../datasets/bayesian_with_clusters.csv', index=False)
    prolog_frame.to_csv('../datasets/prolog_with_clusters.csv', index=False)

    print("Preprocessing done")


main("../datasets/preprocessed.csv", 2, 5)
