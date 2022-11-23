# sal-metaclustering

The software framework explores the evaluation of machine learning (ML) prediction performance in the context of leukemia disease, validate new hypotheses, and extract insights.

## Analysis

Based on a feasibility study conducted in advance, the data of the SAL dataset is preprocessed and different ML models are trained as examples. Thereby, different target metrics are given as learning objectives for Supervised ML, or given as comparison metrics for Unsupervised ML. The investigation, or software framework, therefore focuses classification and cluster analysis techniques from established ML technologies to gain a new view on the data, beyond the usual metrics (scores, e.g. ELN 2017), or to test new theses.

The intended outcome is an exemplary supervised- and unsupervised-ML-based analysis and, in particular, a process that can be adapted and refined by clinicians after the project is completed, providing a starting point for many more data-driven investigations.

In the following sections, the problem described is captured and analyzed. The provided data set is examined, i.e., the shape is captured and relevant target variables and attributes / features are filtered. Subsequently, the features and learning objectives are analyzed individually, i.e. examined for information content, meaningfulness and completeness. These static findings form the basis for further machine-aided analysis.


### Problem
#### Aim of this analysis

The goal is to investigate whether data-driven (ML) approaches allow for equal or better predictions on learning objectives, or provide new insights.

#### Utilization of the results

The findings will be exploited primarily in scientific publications.

#### Type of learning system

The prototype supports experiments on classification, regression, and clustering with dimension reduction. The specific learning goals are addressed by Supervised ML methods. Clustering, in conjunction with dimension reduction, is intended to enable the discovery of new relationships. For this purpose, freely configurable comparison metrics for clusters are provided.

#### Measuring prediction qualities

The quality of the classification algorithms is measured by the following metrics:

    F1-Score

    Precision

    Recall

The quality of the regression algorithms is measured by the RMSE.

The quality of the clustering algorithms is measured by cluster compare metrics and additionally output as general, so-called internal metrics (without ground truth).


### Data set

The hematology department of the UKD provides a data set (SAL data) in the form of a CSV file. The data is pseudonymized and the individual attributes/features are described, or a description in the form of a dictionary was provided by the client.

#### Learning objectives

The learning objectives (labels) are included in the data, specifically:

Overall Survival Time (OSTM, months).

    The time from the date of diagnosis or initiation of treatment until the patient's death.
Relapse-free Survival Time (RFS, months).

    The time after initial treatment for a cancer that the patient survives without signs or symptoms of that cancer.
Event-free Survival Time (EFS, months).

    The time after initial treatment of a cancer in which the patient remains free of certain complications or events that the treatment was intended to prevent or delay.
First Complete Remission (CR1, truth value).

    The disappearance of all signs of cancer in response to treatment.

#### Preprocessing

The CSV file 1 provided contains data without a defined schema. Therefore, the data was imported into a relational MySQL database and provided with a schema via a view (projection). The associated script 2 defines this view, as well as some auxiliary functions for typing and cleaning the data. Specifically, after analyzing the columns, the following processing steps were taken:

* strings with (floating point) numbers were converted
* empty strings were provided with NULL values (missing values)
* each date was converted into a Unix timestamp

The advantage of this form of preprocessing is the interoperability (between Weka, TensorFlow, Scikit-Learn, PyTorch, etc.) of the typed data. In this process, the MySQL Workbench 3 software allows for
* loading the CSV data into a table
* execution of the script to define the projection
* the subsequent export of the projected data as CSV 4

#### Attributes defined as irrelevant

In the preliminary discussion, the following columns of the dataset were excluded for analysis:
* ALSCTCR1
* ALSCTDTC
* ALSCTSLV
* DoubleInduction
* IT1RES
* PatMatID
* PTF
* PTFDTC
* R1
* R1DTC
* R2
* R2DTC
* R3
* R3DTC
* SUBJID
* TIDTC
* TRIALID
* TRT
* TRTIND'

The exclusion of the corresponding attributes is implemented in sal.features.selection.drop_irrelevant_attributes().

### Exploration

In a first step, the individual attributes, irrelevant and constant excluded, of the 3,793 data records were examined. In order to be able to guarantee the unique assignability of a data record, the attribute SUBJID was used as an index.
Attribute analysis

The analysis of the attributes of the SAL dataset is structured according to the data type. Due to the predominant proportion, mutation indicators, as qualitative binary attributes, are described separately. The results are available to the implementation in sal.data.Attrs.

* Qualitative attributes
  * Nominal attributes
    * AMLSTAT
    * CD34SRC
    * WHO
    * CISTAT
    * CEBPASTAT
    * IDH1T
    * IDH2T
    * FAB
  * Ordinal attributes
    * ECOG
    * ELNRisk
    * CGELN
    * CGSTUD
  * Binary attributes
    * EXAML
    * FEV
    * SEX
    * RFSSTAT
    * EFSSTAT
    * OSSTAT
    * CR1
    * ED30
    * ED60
    * CGCX
    * CGNK
    * Mutation indicators
* Quantitative Attributes
  * Discrete attributes
    * AGE
    * PBB
    * BMB
    * D16BMB
* Continuous attributes
    * RFSTM
    * OSTM
    * EFSTM
    * CD34
    * WBC
    * PLT
    * LDH
    * HB
    * FIB
    * FLT3R
* Other attributes
  * Textual attributes
    * CGKT
    * DNMT3AT
  * Date values

#### Information content

The following attributes are redundant and do not add value to ML supported analysis:
* ECOGCAT (redundant to ECOG)
* AGEGR (redundant to AGE)

#### Missing values

In this figure, all attributes are shown on the X axis. The white spaces symbolize missing values. The "sparkline" on the right side stands for known values. What is striking here is the formation of two classes within the entire data set. .. In order not to create a sampling bias, these two groups must be distinguished.

#### ELN Risk

A risk classification according to the European LeukemiaNet (Döhner et al. 2017) is performed by the qualitative, attributes ELNRisk, CGELN and CGSTUD. The specific values represent "favorable" (fav), "unfavorable" (adv), and "intermediate" (int). ELN risk corresponds to a learning target (training exclusion).
#### Relapse-free Survival (RFS).

RFS corresponds to a learning objective (training exclusion).

#### Event-free Survival (EFS)

EFS corresponds to a learning objective (training exclusion).

#### Cytogenetic cariotype

The processing of CGKT is an optional part of the project. The absence of this information can be compensated by a combination of CGCX, CGNK and ELNRisk.



## Integration

Integration includes the description of the entire data processing path, hereafter referred to as the pipeline.
The literature often describes pipelines as consisting of three parts. [#]_ The first part concerns :ref:`Feature Selection`
and/or :ref:`Engineering<Feature Engineering>`
in favor of the smallest and most expressive feature set. In the second one algorithms
of a model family are selected and in the third :term:`Hyper-Parameters` are set in favor of
optimal performance.

In the SAL analysis, a fourth part is introduced for evaluating the results and their presentation.
The actual ML models and meta / hyper models are combined to generalized hierarchical models.

Therefore, the parts / modules provided in this project include:

* Pre-processing of raw data, and selection and preparation of appropriate attributes.

* creation and optimization of ML models, respectively algorithms, i.e. the actual transformation and learning procedures

    * special comparison metrics (:term:`cluster-compare-metrics`) for evaluation of the results

* result evaluation and visualization.

The pipeline used here for supervised-ML procedures (classification and regression) is structured as follows:

* Data import
* Feature Selection / Filtering / Transformation & Scaling
* Model Construction and Optimization (Training)
* Evaluation & Visualization

The pipeline used here for Unsupervised ML methods (clustering) is structured as follows:

* Data import
* Feature Selection / Filtering / Transformation & Scaling
* Model Transformation (Dimensionality Reduction and Unification)
* Model Construction and Optimization (Training)
* Evaluation & Visualization

Thus, the pipelines are similar and therefore share common software components. These are described below:

### Features

The following configurable sub-pipeline generates feature records and variants from the given pre-processed SAL dataset.

The preprocessing of features is divided into the following subsections:


#### Data Cleaning

Data cleaning involves the imputation of missing values and the encoding of existing values. Different data types from the attribute analysis are treated as equally as possible to minimize distortions. The implementation is in the class sal.features.DefaultColumnTransformer.

##### Handling of missing values

As shown earlier in the analysis, the SAL dataset is incomplete. Not all machine learning methods are able to handle this. Since many ML-V methods are to be compared, these missing values are generally imputed. Since imputation can lead to a distortion of the problem, imputed values for the attribute ECOG as well as for all discrete attributes are marked with an additional binary attribute (cf. MissingValuesIndication).

The following imputation of ordinal, discrete and continuous attributes is done with the median of all values of the respective attribute.

All variants of the ELN score (ELNRisk, CGELN and CGSTUD), as well as nominal attributes are imputed with a constant value (missing/unknown/etc.).

##### Coding of categorical attributes.

All nominal attributes, binary attributes, and mutation indicators are transformed by sal.data.encode.BinaryNominalEncoder, where 1 and 0 are replaced by Y and N, and missing, and non-binary values are ignored (cf. sal.data.encode.BinaryNominalEncoder).

The variants of the ELN score are encoded ordinally and all nominal attributes are encoded one-hot.

#### Feature Selection

The number of origin attributes in the SAL dataset is high (up to 138) and can be reduced considerably. Such a reduction should be made to avoid the "curse of dimensionality". 1

One effect includes overfitting with supervised ML, where the model performs well with training data but generalizes poorly with new data. The model then detects patterns in the noise/randomness itself.

Furthermore, clustering is made more difficult with Unsupervsied ML. Too many dimensions cause statistical methods, on which Unsupervised ML is essentially based, to fail, so that elements in the data set appear equally distant from all others. As a result, similarity measures, such as Euclidean distance, yield useless results and clustering quality declines.

Thus, choosing an appropriate subset of features is important not only for speeding up the training of the models, but also for the interpretability of the results.

Feature selection is used in two different places:

##### Static selection

Before the original data is converted, attributes defined as irrelevant, constant attributes (units and the like) and those with low information content are to be filtered out. The implementation is performed by sal.features.selection.DefaultAttributeSelector and is located in sal.features.selection.drop_irrelevant_attributes(), sal.features.selection.drop_constant_attributes() and sal.features.selection.drop_features_with_low_information_gain().

##### Dynamic selection

Once the dataset has been generated by the pipeline, the remaining features are subjected to tests for fitness with respect to a specific target. For example, the correlation between a learning goal such as CR1 and each feature can be calculated. Then, if a certain threshold is exceeded, the algorithm by definition supports the feature with respect to its expressiveness. The following methods are used for this purpose:

* Correlation
* ꭓ²-Test
* Recursive feature elimination
* Lasso regularization
* Random-Forest Ranking

A table is then calculated with the truth value as the product of method and feature. A threshold for the number of supporting algorithms is set in the configuration and decides which features are included in the training.


#### Feature Engineering

Exploration has documented known relationships between the information contained in the SAL dataset. This and other domain knowledge aids in the composition and synthesis of additional features that can be instrumental in improving the performance of the models. No targeted feature engineering was implemented as part of the SAL analysis project and remains open for further analysis steps.

#### Feature Scaling

One of the most important transformations applied to the feature dataset is scaling. Few machine learning algorithms work well when the numberic attributes have very different scales (cf. HB and LDH). The two most common methods to homogenize the scales of all attributes are normalization (min-max scaling) and standardization (Z-score).

From the visualizations of the attribute analysis, we suspected outliers for some attributes:

* PLT
* LDH
* HB
* FIB

Since standardization is less prone to outliers, it is preferred over normalization and applied to all attributes.

The attributes D16BMB and EFSTM can also be partially freed from outliers using a logarithmic transformation. A power transformation can be applied to RFSTM, OSTM, WBC and FLT3R.

### Models

The models and meta-models implemented or used are described below.


#### Supervised

After the feature dataset is fixed, the following steps are processed by the scripts for Supervised ML models. The implementation is in the methods sal.models.supervised.conduct_classification_experiment() and sal.models.supervised.conduct_regression_experiment().

1. Loading a variant of the features
2. Removing datasets with unknown learning target (these are stored under data/processed/<learning target>_unknown.csv)
3. Splitting the data into training and test sets
4. Scaling of the data
5. Selection of the optimal feature set
6. for all models:
    * Perform ten-fold cross-validation 1
    * Storing performance by test set in numbers and visualizations

The differences between classification and regression algorithms are described below.

##### Classification models

For classification problems, stratified randomization is used in step three (see above) because the number of data sets differs greatly in the classes to be predicted (unbalanced data). Stratified randomization is also used in cross-validation in step six. The performance is mainly measured by the F1 score and is visible in four different visualizations.

* Algorithms available in the configuration for classification problems:
* GaussianNB: Gaussian Naive Bayes
* LogisticRegression: Logistic Regression
* XGBClassifier: Gradient Boosting
* KNeighborsClassifier: k-nearest Neighbors
* DecisionTreeClassifier: Decision Tree
* RandomForestClassifier: Random Forest
* MLPClassifier: Multi-Layer Perceptron
* AdaBoostClassifier: AdaBoost
* RidgeClassifier: Ridge
* SVC: Support Vector
* NuSVC: Nu-Support Vector
* LinearSVC: Linear Support Vector
* SGDClassifier: Linear Classifiers with Stochastic Gradient Decent
* PassiveAggressiveClassifier: Passive Aggressive
* QuadraticDiscriminantAnalysis: Quadratic Discriminant Analysis

###### Class Imbalance

In addition to stratified randomization, other approaches can be used to balance the data:

* Downsampling (reduce number of majority class records).
* Upsampling with SMOTE algorithm 2

##### Regression models

For regression problems, simple randomization is used in step three. The performance is mainly measured by the RMSE and is visible in three different visualizations.
Algorithms for regression problems available in the configuration:

* LinearRegression: Linear Regression
* Ridge: Linear Regression with L2 Regularization
* Lasso: Linear Model trained with L1 prior as regularizer
* ElasticNet: Linear Regression with combined L1 and L2 priors as regularizer
* Lars: Least Angle Regression
* OrthogonalMatchingPursuit: Orthogonal Matching Pursuit
* BayesianRidge: Bayesian ridge regression
* ARDRegression: Bayesian ARD Regression
* SGDRegressor: Linear Model fitted by minimizing a regularized empirical loss with SGD
* PassiveAggressiveRegressor: Passive Aggressive Regression
* RANSACRegressor: RANSAC (RANdom SAmple Consensus) algorithm
* TheilSenRegressor: Theil-Sen Estimator: robust multivariate regression
* HuberRegressor: Linear Regression Model that is robust to outliers
* DecisionTreeRegressor: Decision Tree Regression
* GaussianProcessRegressor: Gaussian Process Regression
* MLPRegressor: Multi-Layer Perceptron Regression
* KNeighborsRegressor: k-nearest Neighbors Regression
* RadiusNeighborsRegressor: Regression based on Neighbors within a fixed radius
* SVR: Support Vectors
* NuSVR: Nu-Support Vectors
* LinearSVR: Linear Support Vector Regression


#### Unsupervised
##### Data transformation
###### Description

To enable Unsupervised ML, here cluster analysis, an additional preprocessing step is necessary to reduce and unify highly dimensional and heterogeneous data, like the SAL dataset. This is done by transferring the individual data elements into a new model space in which the (Euclidean) distances represent the original similarity of the features (dimension reduction). This makes the data accessible to most clustering procedures in the first place.

Through these procedures, the features of the data elements are completely replaced by uniform, scaled features. The mapping holds:

(f_1, .., f_n) -> (g_1, .., g_m), where n >> m.

The methods thereby aim to preserve the similarity of data elements described by their features. This step significantly reduces redundancy, bias due to different scales and data types, but potentially changes the data set significantly. Which methods are target-oriented can only be evaluated by systematic testing. The mapping from the original to the reduced space is maintained, so that a reverse transformation is possible.

###### Control

There exists a large number of different procedures (dimension reduction), which have been integrated to a large extent.

The transformation parameter describes a list of transformation procedures to be executed in the grid_search.yml configuration.
available methods for dimension reduction:


* Principal Component Analysis: PCA
* Incremental Principal Component Analysis: Incr. PCA
* Sparse Principal Component Analysis: Sparse PCA
* Singular Value Decomposition: SVD
* Gaussian Random Projection: GRP
* Sparse Random Projection: SRP
* Multi Dimensional Scaling: MDS
* ISOMAP: ISOMAP
* Linear Local Embedding: LLE
* Mini-Batch Dictionary Learning: MBDL
* AutoEncoder: AutoEncoder


The most important parameter is the number of targeted uniform dimensions in the image space, as this significantly affects the shape and expressiveness of the intermediate model. A high number allows for good mapping but makes it difficult for subsequent clustering methods to detect good clusters; a low number potentially distorts the representation but allows for good clustering.

Additional, individual parameters for individual procedures can improve the results, setting them via configuration can be retrofitted.

The target_dimensionalities parameter in the grid_search.yml configuration describes a list of dimension numbers to be examined. Common values are between 2 and 8.

##### Clustering procedure

On the basis of the transformed models the actual clustering takes place, i.e. the independent abstraction as a division into different groups. The clustering methods follow different strategies, but for good clustering general criteria apply:

* balanced number of clusters (no too strong or too weak abstraction)
* large differences between the clusters (the elements of different clusters should differ strongly in terms of their features)
* small differences within clusters (the elements within a cluster should differ little in terms of their features).

A large number of different clustering methods exist (overview) whereby the suitability of certain methods can only be determined by systematic testing. A large number of procedures have been included, more are possible.
available clustering methods

* k-Means: k-Means
* Two Means: mbk-Means
* Agglomerative Clustering: AC
* Spectral Clustering: SC
* Linkage: Linkage
* BIRCH: BIRCH
* Gaussian Mixture: GM
* OPTICS: OPTICS
* MeanShift: MeanShift
* DBSCAN: DBSCAN
* Affinity Propagation: AffProp


The parameter clustering describes in the configuration grid_search.yml a list of clustering procedures to be executed.

Basically, the procedures differ in whether a target cluster count must be specified or whether the procedures determine this themselves. Both procedures have advantages and disadvantages, so both groups were included in the prototype.

The parameter target_number_of_clusters describes in the configuration grid_search.yml a list of cluster numbers to be examined for the procedures that require an explicit specification.

To ensure the quality of the clusters, a simple sanity check is performed after clustering. This is defined by parameters min_number_of_clusters, max_number_of_clusters and min_rel_cluster_size in the configuration grid_search.yml . The parameters describe the minimum and maximum number of clusters, as well as the rel. minimum size of each cluster to avoid extreme imbalances. After a failed sanity check, the result is discarded and not evaluated further.

The result of all clustering procedures is the assignment of the individual data elements from the transformed model to a group. Fuzzy procedures are not considered.

##### Clustering Evaluation
###### Specific cluster difference metric

In order to evaluate the overall quality of data selection, transformation, and clustering, a specific metric is defined that serves as a measure of the quality of the results, in addition to the general so-called internal cluster qualities (i.e., those without ground truth):

The features of a cluster are defined by the features of the assigned elements aggregated by a statistical function (e.g., mean or median)
[featcl]=[funcstat(featelement1,featelement2)]

The difference between two clusters with respect to a feature is a difference function (e.g. relative difference, absolute difference) weighted by a global scaling function scal_glob(value) to evaluate differences in different number ranges differently. value is a user-defined parameter and should be derived from the feature values.
difffeat=funcdiff(featcl1,featcl2)∗scalglob(value)

The difference between two clusters defines a statistical function (e.g. mean, median) over the entire vector feature differences.
diffcl=funcstatcl([difffeat])

The total difference between all clusters is calculated from the weighted (cluster size) mean of all pairwise cluster differences
difftotal=meanweighted([diffcl],[sizecl1+sizecl2])

The statistical functions, as well as the difference functions and the global scaling functions can be exchanged, or extended. See sal.models.cluster_diff_evaluation.py

In the performed investigation, 2 comparison metrics are provided as examples:

* Saturated_Pairwise_Mean
  * Mean for feature differences
  * Saturation function for global scaling
  * Mean for cluster difference
* Pairwise_Median
  * Median for feature differences
  * Constant (1) for global scaling
  * Mean for cluster difference

Due to this free definition of the cluster comparison, or clustering quality, the results are only meaningful in comparison.

###### Internal cluster qualities

Since no reference, i.e. ground truth is available due to the tasks (extraction of new knowledge), the internal cluster quality measures are severely limited. The measures implemented were (description).

* Silhouette coefficient
* Calinski-Harabasz index
* Davies-Bouldin score

The higher a measure is, the better the quality of the cluster is considered to be (separated, density distribution, ...).

##### Grid search as a meta-model

Since the intermediate results, i.e. the transformed data model and the clustering cannot be assessed, or the final results of a run can only be assessed in comparison, a systematic and targeted search must be performed. All results are collected and ordered with respect to a metric and output at the end.

A so-called grid search tests a large number of experiments, thus an optimal coverage is achieved at the expense of runtime.

In the prototype, the following model for a grid search is implemented:

* Execute all runs
  * Tests all given transformation methods
    * Tests all given target dimension numbers
      * Tests all given clustering methods
        * Test all given target cluster numbers
          * Evaluate cluster result (internal and specific metrics)
            * Save result of experiment

* Sort results according to specific cluster metrics
* Create overviews
* Save overall and individual results

The max_number_of_results parameter in the grid_search.yml configuration describes the number of individual results to store.

The random_seed parameter in the grid_search.yml configuration defines a starting value to make the behavior of pseudo-stochastic methods repeatable.

The runs_per_config parameter in the grid_search.yml configuration defines the number of total runs of the grid search. This is useful to investigate the influence of (pseudo) randomness.

Each degree of freedom of the grid search increases the search space by a factor, so such a search must be well planned.


### Visualization

The visualizations generated by the execution are described below.

#### Supervised Learning
##### Classification

The following visualizations are used to support the evaluation of classifiers.

###### Confusion matrix

This table compares predicted classes (Predicted Class) with the actual ones (True Class). This makes visible to what absolute extent the classifier has decided correctly or incorrectly.

###### Classification Report

The classification report presents the metrics Precision, Recall, and their weighted harmonic mean (F1 score) for the individual classes. The example above shows that 78% of the data sets classified with "N" are actually assigned to this class. Furthermore, it shows that 90% of the records classified with "N" were found by the classifier.

###### Precision-Recall-Curve

The graph shows the trade-off between the accuracy (Precision), and the completeness (Recall) of a classifier. The average accuracy (Avg. Precision) expresses the curve in a single number representing the area under the curve. When choosing a classifier, an attempt is made to maximize both accuracy and completeness.

###### ROC curve

A Receiver Operating Characteristic (ROC) graph allows the user to visualize the trade-off between the sensitivity and specificity of the classifier. Receiver Operating Characteristic (ROC) is a measure of the predictive quality of a classifier. The graph shows the rate of true positives on the Y-axis and the rate of false positives on the X-axis, both as a global average and per class. The ideal point is the upper left corner of the graph.

Another metric is provided by the area under the curve (AUC), which is a quantification of the relationship between false positives and actual positives. In general, the higher the AUC, the better the model.

##### Regression

When evaluating regressors, the following visualizations are used in a supportive manner.

###### Residual diagram

Residuals describe the difference between the observed value of the target variable (y) and the predicted value (ŷ), i.e., the error of the prediction. The errors are normally distributed for a fitted model. If patterns are apparent, the model must be adjusted.

###### Prediction error

A prediction error plot shows the observed values of the target variable (y) from the data set against the predicted values generated by our model (ŷ). This reveals the variance present in the model. The coefficient of determination (R²) describes the goodness of fit of the model.

###### Learning curve

The learning curve describes a relationship between the training and cross-validation metrics (e.g. RMSE) with respect to a variable number of training examples. Among other things, it can be shown how much the model benefits from larger data sets - convergence with respect to fitting is achieved.

The curves are plotted with the mean values. Variability during cross-validation is indicated by the shaded areas representing the standard deviations above and below the mean for all cross-validations. If the model suffers from errors due to bias, the variability around the training curve is likely to be larger. If the model has errors due to variance, there is greater variability around the cross-validated score.


### Unsupervised Learning

Various visualizations and charts are used to present the results.

#### Experiment overview

* X-axis: name of the experiment
  * Number of the run
  * Transformation method
  * Target dimensionality
  * Cluster procedure
  * Target cluster count
  * Cluster comparison metric
* Y-axis left: Cluster comparison metric
* Y-axis right: relative internal cluster qualities.

The experiments are ordered according to the spec. cluster difference, i.e., the experiments on the left are the most promising in terms of the metric. The general qualities serve as orientation from a general perspective on the clusters.

#### Visualization transformation

* Title: Experiment name
  * Number of the run
  * Transformation method
  * Target dimensionality 
  * Cluster procedure
  * Target cluster count
  * Cluster comparison metric
* Subplot:
  * Sub-title: two selected dimensions, corresponds to viewing "direction".
  * X / Y: uniform axes of the image space
  * each point corresponds to a data item, distances between points correspond to feature similarities in the original space

#### Visualization clustering

* Title: Experiment name
    * Number of the run
    * Transformation method
    * Target dimensionality
    * Cluster procedure
    * Target cluster count
    * Cluster comparison metric
* Subplot:
    * Sub-title: two selected dimensions, corresponds to viewing "direction".
    * X / Y: uniform axes of the image space
    * each colored dot corresponds to a data item associated with a cluster

##### Visualization cluster comparison

* Title: Experiment name
    * Number of the run
    * Transformation method
    * Target dimensionality
    * Cluster procedure
    * Target cluster count
    * Cluster comparison metric
* Subplot:
    * Sub-title: cluster pair for comparison
    * X: listing of features, ordered by cluster difference
    * Y: difference of individual features according to spec. metric
    * Blue: features with influence on clustering
    * Red: features without influence (learning goals)

Thus, features with large differences between clusters are interesting candidates for potential new scores or cluster criteria.

#### Visualization of cluster differences with respect to comparison features

* X-axis: clusters with features for comparison (without influence on clustering, learning objectives).
* Y-axis: mean values per cluster and feature (note: Z-normalized for better comparability).

So this chart visualizes the differences of the clusters with respect to the comparison features, i.e. the red features from the cluster comparison.

