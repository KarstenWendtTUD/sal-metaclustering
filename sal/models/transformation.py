#region imports
import numpy as np

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, IncrementalPCA, SparsePCA, MiniBatchDictionaryLearning, \
    FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from numpy.random import seed
import tensorflow as tf
#endregion


#region base class



class SALA_DataTransformer:
    name = ""
    target_dim_count = 2
    console_output = False
    random_state = 0

    def __init__(self, name = "not set", console_output = False):
        self.name = name
        self.console_output = console_output

    def run(self, target_dim_count, data, random_state):
        self.target_dim_count = target_dim_count
        self.random_state = random_state

        if self.console_output:
            print("Start " + self.name)
            print("  Target Dim Count: " + str(target_dim_count))

        npa_trans = self.fit_transform(data)

        #if self.show_scatter_plot:
        #    self.scatter_plot_transposed_data(npa_trans, self.target_dim_count)
        #    if self.target_dim_count == 3:
        #        self.scatter_plot_transposed_data_3d(npa_trans)

        if self.console_output:
            print("End " + self.name)

        return npa_trans

    #def scatter_plot_transposed_data(self, data, dim_count):
    #    for d1 in range(0, dim_count - 1):
    #        for d2 in range(d1 + 1, dim_count - 0):
    #            plt.figure(figsize=(16, 10))
    #            plt.scatter(data[:, d1], data[:, d2], alpha=0.3)
    #            plt.show()

    #def scatter_plot_transposed_data_3d(self, data):
    #    fig = plt.figure(figsize=(16, 10))
    #    ax = fig.add_subplot(111, projection='3d')
    #    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.1, depthshade=False)
    #    plt.show()

    def fit_transform(self, target_dim_count, data):
        pass
#endregion


#region transformers
# principal component analysis (PCA)
class SALA_PCA(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "PCA", console_output)

    def fit_transform(self, data):
        transformation = PCA(n_components = self.target_dim_count,
                             random_state=self.random_state)
        npa_trans = transformation.fit_transform(data)

        if self.console_output:
            print("Explained Variance:")
            explained_variance = transformation.explained_variance_ratio_
            print(explained_variance)

        return npa_trans

# Incremental Principal Component Analysis
class SALA_IncrementalPCA(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "IncrPCA", console_output)

    def fit_transform(self, data):
        transformation = IncrementalPCA(n_components = self.target_dim_count)

        n_batches = len(data) / 8

        for X_batch in np.array_split(data, n_batches):
            transformation.partial_fit(X_batch)
        npa_trans = transformation.transform(data)

        if self.console_output:
            print("Explained Variance:")
            explained_variance = transformation.explained_variance_ratio_
            print(explained_variance)

        return npa_trans

# Kernel Principal Component Analysis
class SALA_KernelPCA(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "KernelPCA", console_output)

    def fit_transform(self, data):
        transformation = KernelPCA(kernel="rbf",
                                   fit_inverse_transform=True,
                                   gamma=10,
                                   random_state=self.random_state)  # rbf = radial basis function
        df_temp = transformation.fit_transform(data)

        pca = PCA(n_components=self.target_dim_count)
        npa_trans = pca.fit_transform(df_temp)

        if self.console_output:
            print("Explained Variance:")
            explained_variance = pca.explained_variance_ratio_
            print(explained_variance)

        return npa_trans

# Sparse Principal Component Analysis
class SALA_SparsePCA(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "SparsePCA", console_output)

    def fit_transform(self, data):
        transformation = SparsePCA(n_components = self.target_dim_count,
                                   random_state=self.random_state)
        npa_trans = transformation.fit_transform(data)

        return npa_trans

# Remark: not working
# Linear Discriminat Analysis
class SALA_LDA(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "LDA", console_output)

    def fit_transform(self, data):
        transformation = LinearDiscriminantAnalysis(n_components=self.target_dim_count)
        npa_trans = transformation.fit(data).transform((data))

        return npa_trans

# Singular Value Decomposition
class SALA_SVD(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "SVD", console_output)

    def fit_transform(self, data):
        transformation = TruncatedSVD(n_components=self.target_dim_count,
                                      random_state=self.random_state)
        npa_trans = transformation.fit_transform(data)

        return npa_trans

# Gaussian Random Projection
class SALA_GRP(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "GRP", console_output)

    def fit_transform(self, data):
        transformation = GaussianRandomProjection(n_components=self.target_dim_count,
                                                  random_state=self.random_state)
        npa_trans = transformation.fit_transform(data)

        return npa_trans

# Sparse Random Projection
class SALA_SRP(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "SRP", console_output)

    def fit_transform(self, data):
        transformation = SparseRandomProjection(n_components=self.target_dim_count,
                                                random_state=self.random_state)
        npa_trans = transformation.fit_transform(data)

        return npa_trans

# Multi-Dimensional Scaling
class SALA_MDS(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "MDS", console_output)

    def fit_transform(self, data):
        transformation = MDS(n_components=self.target_dim_count,
                             random_state=self.random_state)
        npa_trans = transformation.fit_transform(data)

        return npa_trans

# T-distributed Stochastic Neighbor Embedding
class SALA_tSNE(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "t-SNE", console_output)

    def fit_transform(self, data):
        tsne_perplexity = 20

        if self.target_dim_count > 3:
            return None

        npa_trans = TSNE(
            learning_rate=100,
            n_components=self.target_dim_count,
            perplexity=tsne_perplexity,
            init="random",
            method="barnes_hut",
            random_state=self.random_state
        ).fit_transform(data)

        return npa_trans

# ISOMAP algorithm
class SALA_Isomap(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "ISOMAP", console_output)

    def fit_transform(self, data):
        transformation = Isomap(n_components=self.target_dim_count)
        npa_trans = transformation.fit(data).transform(data)

        return npa_trans

# Linear Local Embedding
class SALA_LLE(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "LLE", console_output)

    def fit_transform(self, data):
        transformation = LocallyLinearEmbedding(
            # method = "modified",
            eigen_solver="dense",
            n_components=self.target_dim_count,
            random_state=self.random_state)
        npa_trans = transformation.fit(data).transform(data)

        return npa_trans

# Mini-Batch Dictionary Learning
class SALA_MBDL(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "MBDL", console_output)

    def fit_transform(self, data):
        transformation = MiniBatchDictionaryLearning(n_components=self.target_dim_count,
                                                     random_state=self.random_state)
        npa_trans = transformation.fit_transform(data)

        return npa_trans

# not working
# Independent Composent Analysis
class SALA_ICA(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "ICA", console_output)

    def fit_transform(self, data):
        transformation = FastICA(n_components=self.target_dim_count)
        npa_trans = transformation.fit_transform(data)

        return npa_trans

# Neural Network Autoencoder
class SALA_AutoEncoder(SALA_DataTransformer):
    def __init__(self, console_output = False):
        SALA_DataTransformer.__init__(self, "AutoEnc", console_output)

    def fit_transform(self, data):
        seed(self.random_state)
        #tf.set_random_seed(self.random_state)

        m = Sequential()
        m.add(Dense(512, activation='elu', input_shape=(len(data.columns),)))
        m.add(Dense(128, activation='elu'))
        m.add(Dense(self.target_dim_count, activation='linear', name="bottleneck"))
        m.add(Dense(128, activation='elu'))
        m.add(Dense(512, activation='elu'))
        m.add(Dense(len(data.columns), activation='sigmoid'))
        m.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.005))
        history = m.fit(data, data, batch_size=128, epochs=100, verbose=1)

        encoder = Model(m.input, m.get_layer('bottleneck').output)
        Zenc = encoder.predict(data)
        Renc = m.predict(data)

        return Zenc

#endregion
