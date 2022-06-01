"""
Alle Funktionen/Module die das Training von Modellen und die Vorhersage betreffen.
"""
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, \
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, \
    SGDRegressor, PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

from sal.models.cluster_diff_evaluation import *
from sal.models.clustering import *
from sal.models.transformation import *

#: Mapping vom Namen eines Modells auf eine Funktion,
#: die eine Instanz erstellt. Wird für die YAML-Konfiguration benötigt.
model_factory_mapping = {
    # Klassifikation
    'GaussianNB': lambda: GaussianNB(),
    'LogisticRegression': lambda: LogisticRegression(),
    'XGBClassifier': lambda: XGBClassifier(),
    'KNeighborsClassifier': lambda: KNeighborsClassifier(),
    'DecisionTreeClassifier': lambda: DecisionTreeClassifier(),
    'RandomForestClassifier': lambda: RandomForestClassifier(),
    'MLPClassifier': lambda: MLPClassifier(),
    'AdaBoostClassifier': lambda: AdaBoostClassifier(),
    'RidgeClassifier': lambda: RidgeClassifier(),
    'SVC': lambda: SVC(),
    'NuSVC': lambda: NuSVC(),
    'LinearSVC': lambda: LinearSVC(),
    'SGDClassifier': lambda: SGDClassifier(),
    'PassiveAggressiveClassifier': lambda: PassiveAggressiveClassifier(),
    'QuadraticDiscriminantAnalysis': lambda: QuadraticDiscriminantAnalysis(),

    # Regression
    'LinearRegression': lambda: LinearRegression(),
    'Ridge': lambda: Ridge(),
    'Lasso': lambda: Lasso(),
    'ElasticNet': lambda: ElasticNet(),
    'Lars': lambda: Lars(),
    'OrthogonalMatchingPursuit': lambda: OrthogonalMatchingPursuit(),
    'BayesianRidge': lambda: BayesianRidge(),
    'ARDRegression': lambda: ARDRegression(),
    'SGDRegressor': lambda: SGDRegressor(),
    'PassiveAggressiveRegressor': lambda: PassiveAggressiveRegressor(),
    'RANSACRegressor': lambda: RANSACRegressor(),
    'TheilSenRegressor': lambda: TheilSenRegressor(),
    'HuberRegressor': lambda: HuberRegressor(),
    'DecisionTreeRegressor': lambda: DecisionTreeRegressor(),
    'GaussianProcessRegressor': lambda: GaussianProcessRegressor(),
    'MLPRegressor': lambda: MLPRegressor(),
    'KNeighborsRegressor': lambda: KNeighborsRegressor(),
    'RadiusNeighborsRegressor': lambda: RadiusNeighborsRegressor(),
    'SVR': lambda: SVR(),
    'NuSVR': lambda: NuSVR(),
    'LinearSVR': lambda: LinearSVR(),

    # Dimensionalitätsreduktion
    'PrincipalComponentAnalysis': lambda: SALA_PCA(),
    'IncrementalPrincipalComponentAnalysis': lambda: SALA_IncrementalPCA(),
    'KernelPrincipalComponentAnalysis': lambda: SALA_KernelPCA(),
    'SparsePrincipalComponentAnalysis': lambda: SALA_SparsePCA(),
    'SingularValueDecomposition': lambda: SALA_SVD(),
    'GaussianRandomProjection': lambda: SALA_GRP(),
    'SparseRandomProjection': lambda: SALA_SRP(),
    'MultiDimensionalScaling': lambda: SALA_MDS(),
    't-StochasticNeighborEmbedding': lambda: SALA_tSNE(),
    'ISOMAP': lambda: SALA_Isomap(),
    'LinearLocalEmbedding': lambda: SALA_LLE(),
    'Mini-BatchDictionaryLearning': lambda: SALA_MBDL(),
    'AutoEncoder': lambda: SALA_AutoEncoder(),

    # Clustering
    'k-Means': lambda: SALA_kMeans(),
    'TwoMeans': lambda: SALA_TwoMeans(),
    'AgglomerativeClustering': lambda: SALA_AgglomerativeClustering(),
    'SpectralClustering': lambda: SALA_SpectralClustering(),
    'Linkage': lambda: SALA_Linkage(),
    'BIRCH': lambda: SALA_Birch(),
    'GaussianMixture': lambda: SALA_GaussianMixture(),

    'OPTICS': lambda: SALA_OPTICS(),
    'MeanShift': lambda: SALA_MeanShift(),
    'DBSCAN': lambda: SALA_DBSCAN(),
    'AffinityPropagation': lambda: SALA_AffinityPropagation(),

    'Saturated_Pairwise_Mean': lambda: SALA_Saturated_Pairwise_Mean(),
    'Pairwise_Median': lambda: SALA_Pairwise_Median(),
    'StagedClusterEvaluator1': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator2': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator3': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator4': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator5': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator6': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator7': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator8': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator9': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator10': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator11': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator12': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator13': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator14': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator15': lambda: SALA_StagedClusterEvaluator(),
    'StagedClusterEvaluator16': lambda: SALA_StagedClusterEvaluator()
}


def map_mlp_parameters(params: dict):
    n_neurons = params['n_neurons_per_layer']
    n_layers = params['n_hidden_layer']

    # create the hidden layers as a tuple with length n_layers and n_neurons per layer
    params['hidden_layer_sizes'] = (n_neurons,) * n_layers

    # the parameters are deleted to avoid an error from the MLPRegressor
    params.pop('n_neurons_per_layer')
    params.pop('n_hidden_layer')
