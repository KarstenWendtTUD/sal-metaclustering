#region imports
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, mixture
#endregion


#region base class
class SALA_Clusterer():
    name = ""
    console_output = False
    quantile = .3
    eps = .3
    damping = .9
    preference = -200
    n_neighbors = 10
    n_clusters = 3
    min_samples = 20
    xi = 0.05
    min_cluster_size = 0.1
    random_state = 0

    connectivity = None
    bandwidth = None

    def __init__(self, name = "not set", console_output = False):
        self.name = name
        self.console_output = console_output

    def run(self, target_cluster_count, data, random_state):
        data_dim = data.shape[1]
        self.n_clusters = target_cluster_count
        self.random_state = random_state

        if self.console_output:
            print("Start " + self.name)
            print("  Target Cluster Count: " + str(target_cluster_count))

        X = data.copy()

        # normalize dataset for easier parameter selection
        # X = StandardScaler().fit_transform(X)


        # connectivity matrix for structured Ward
        self.connectivity = kneighbors_graph(X, n_neighbors = self.n_neighbors, include_self=False)
        # make connectivity symmetric
        self.connectivity = 0.5 * (self.connectivity + self.connectivity.T)

        cluster_ids = self.fit_transform(data)

        #if self.show_scatter_plot:
        #    self.scatter_plot_clustered_data(data, cluster_ids, data_dim)
        #    if data_dim == 3:
        #        self.scatter_plot_clustered_data_3d(data, cluster_ids)

        if self.console_output:
            print("End " + self.name)

        return cluster_ids


    def fit_transform(self, data):
        pass


    def get_cluster_ids(self, algorithm, data):
        if hasattr(algorithm, 'labels_'):
            cluster_ids = algorithm.labels_.astype(np.int)
        else:
            cluster_ids = algorithm.predict(data)

        return cluster_ids

    def requires_number_of_clusters(self):
        pass
#endregion


#region clusterers
class SALA_kMeans(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "k-Means", console_output)

    def requires_number_of_clusters(self):
        return True

    def fit_transform(self, data):
        algorithm = cluster.KMeans(n_clusters=self.n_clusters,
                                   random_state=self.random_state)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


# auto detection of number of clusters
class SALA_OPTICS(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "OPTICS", console_output)

    def requires_number_of_clusters(self):
        return False

    def fit_transform(self, data):
        algorithm = cluster.OPTICS(min_samples = self.min_samples,
                                   xi = self.xi,
                                   min_cluster_size = self.min_cluster_size,
                                   eps=self.eps)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


# auto detection of number of clusters
class SALA_MeanShift(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "Mean-Shift", console_output)

    def requires_number_of_clusters(self):
        return False

    def fit_transform(self, data):
        self.bandwidth = cluster.estimate_bandwidth(data, quantile=self.quantile)

        algorithm = cluster.MeanShift(bandwidth=self.bandwidth,
                                      bin_seeding=True)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


class SALA_TwoMeans(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "Two-Means", console_output)

    def requires_number_of_clusters(self):
        return True

    def fit_transform(self, data):
        algorithm = cluster.MiniBatchKMeans(n_clusters= self.n_clusters,
                                            random_state=self.random_state)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


class SALA_AgglomerativeClustering(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "AgglomerativeClustering", console_output)

    def requires_number_of_clusters(self):
        return True

    def fit_transform(self, data):
        if self.n_clusters != None:
            algorithm = cluster.AgglomerativeClustering(n_clusters=self.n_clusters,
                                                        linkage='ward',
                                                        connectivity=self.connectivity)
        else:
            algorithm = cluster.AgglomerativeClustering(n_clusters=None,
                                                        linkage='ward',
                                                        compute_full_tree=True,
                                                        distance_threshold=self.eps)

        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


class SALA_SpectralClustering(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "SpectralClustering", console_output)

    def requires_number_of_clusters(self):
        return True

    def fit_transform(self, data):
        algorithm = cluster.SpectralClustering(n_clusters=self.n_clusters,
                                               eigen_solver='arpack',
                                               affinity="nearest_neighbors",
                                               random_state=self.random_state)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


# auto detection of number of clusters
class SALA_DBSCAN(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "DBSCAN", console_output)

    def requires_number_of_clusters(self):
        return False

    def fit_transform(self, data):
        algorithm = cluster.DBSCAN(eps=self.eps)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


# auto detection of number of clusters
class SALA_AffinityPropagation(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "AffinityPropagation", console_output)

    def requires_number_of_clusters(self):
        return False

    def fit_transform(self, data):
        algorithm = cluster.AffinityPropagation(damping=self.damping,
                                                preference=self.preference,
                                                random_state=self.random_state)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


class SALA_Linkage(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "Linkage", console_output)

    def requires_number_of_clusters(self):
        return True

    def fit_transform(self, data):
        algorithm = cluster.AgglomerativeClustering(linkage="average",
                                                    affinity="cityblock",
                                                    n_clusters=self.n_clusters,
                                                    connectivity=self.connectivity)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


class SALA_Birch(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "BIRCH", console_output)

    def requires_number_of_clusters(self):
        return True

    def fit_transform(self, data):
        algorithm = cluster.Birch(n_clusters=self.n_clusters)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)


class SALA_GaussianMixture(SALA_Clusterer):
    def __init__(self, console_output = False):
        SALA_Clusterer.__init__(self, "GaussianMixture", console_output)

    def requires_number_of_clusters(self):
        return True

    def fit_transform(self, data):
        algorithm = mixture.GaussianMixture(n_components=self.n_clusters,
                                            covariance_type='full',
                                            random_state=self.random_state)
        algorithm.fit(data)
        return self.get_cluster_ids(algorithm, data)
#endregion