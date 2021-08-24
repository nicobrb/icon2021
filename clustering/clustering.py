from os import path
from sys import argv
import pandas as pd
from sklearn_extra.cluster import KMedoids
from matplotlib import pyplot as plt
import seaborn as sns


def k_Medoids(dataframe, max_iters, n_cluster):

    print(str(n_cluster)+"-Medoids clustering...")
    cluster = KMedoids(n_clusters=n_cluster, metric='euclidean',
                       init='build', method='alternate', max_iter=max_iters).fit(dataframe)

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


def main():

    try:
        if not path.isfile(argv[1]):
            print("file not found or wrong directory, returning")
            return

    except IndexError:
        print("Error with number of arguments")
        return

    try:
        if int(argv[2]) > 0:
            max_iter = int(argv[2])
        else:
            print("incorrect iterations value")
            return
    except IndexError:
        max_iter = 10
    try:
        if int(argv[3]) > 0:
            k = int(argv[3])
        else:
            print("incorrect k value for elbow plotting")
            return
    except IndexError:
        k = 5

    dataframe = pd.read_csv(argv[1])

    # elbow(dataframe, max_iter, k)

    cluster = k_Medoids(dataframe, max_iter, k)
    k_medoids_cluster = cluster.labels_

    bay_frame = pd.read_csv('../datasets/bayesian_ready.csv').assign(n_cluster=k_medoids_cluster)
    prolog_frame = pd.read_csv('../datasets/prolog_ready.csv').assign(n_cluster=k_medoids_cluster)

    bay_frame.to_csv('../datasets/bayesian_with_clusters.csv', index=False)
    prolog_frame.to_csv('../datasets/prolog_with_clusters.csv', index=False)

    print("Preprocessing done")


main()

