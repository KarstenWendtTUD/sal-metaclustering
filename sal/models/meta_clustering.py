import logging
import pandas as pd
import numpy as np
import yaml
import os
import datetime

import matplotlib.pyplot as plt

from sal.data.load import typed_view
from sal.models import model_factory_mapping
from sal.models.unsupervised import ExperimentResult, prepare_clustered_data


def start(config_file_path: str):
    logger = logging.getLogger(__name__)

    ##########################
    ## LOAD CONFIGS
    ##########################

    with open(config_file_path) as f:
        logger.info('config loaded from "{}"'.format(config_file_path))
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config is None:
        return False
    random_state = config["random_seed"]

    with open(config["supervised_ML_config"]) as f:
        logger.info('UL config loaded from "{}"'.format(config["supervised_ML_config"]))
        config_UL = yaml.load(f, Loader=yaml.FullLoader)
        config_UL = config_UL["baseline"]
    if config_UL is None:
        return False

    ###############################
    ## LOAD DATA
    ###############################

    df_org_element2clusters = pd.read_csv(config["clusterings"], sep=';')
    logger.info('cluster sets loaded from "{}"'.format(config["supervised_ML_config"]))
    # print(df_org_element2clusters)

    df_org_cluster_qualities = pd.read_csv(config["qualities"], sep=';')
    logger.info('cluster qualities loaded from "{}"'.format(config["qualities"]))
    # print(df_org_cluster_qualities)

    df_org_elements = typed_view(config["elements"])
    logger.info('original element data loaded from "{}"'.format(config["elements"]))
    # print(df_org_elements)

    ################################
    ## PREPARE DATA
    ################################

    # remove subjID as feature
    df_clusterings = df_org_element2clusters.drop(["SUBJID"], axis='columns')

    # filter clusterings of low quality
    filter_name = config["filter_name"]
    min_filter_threshold = config["min_filter_threshold"]

    logger.info('filter clusterings regarding "{}" >= "{}'.format(filter_name, min_filter_threshold))

    for i in range(df_org_cluster_qualities.shape[0]):
        quality_entry = df_org_cluster_qualities.iloc[i]
        if quality_entry[filter_name] < min_filter_threshold:
            df_clusterings.drop(str(i), axis='columns', inplace=True)

    logger.info('clusterings left: {} of {}'.format(df_clusterings.shape[1], df_org_cluster_qualities.shape[0]))

    ####################################
    ## ENCODING
    ####################################



    if True:
        logger.info('encoding cluster IDs')
        #print(df_clusterings)
        df_clusterings_enc = pd.get_dummies(df_clusterings, columns=df_clusterings.columns.tolist())
        #print(df_clusterings_enc)
        df_clusterings = df_clusterings_enc

    ####################################
    ## TRANSFORMATION / DIM REDUCTION
    ####################################

    target_dimensionality = config["target_dimensionality"]
    transformation_name = config["transformation"]
    transformer = model_factory_mapping[transformation_name]()

    logger.info('transformation clusterings data via {} to {} dimensionalities'.format(transformation_name, target_dimensionality))

    # the catches are required because of some reported bug in the numpy svd module
    npa_clusterings_reduced = None
    try:
        random_state += 1
        npa_clusterings_reduced = transformer.run(target_dimensionality, df_clusterings, random_state)
    except np.linalg.LinAlgError as e:
        logger.error(e)
    except ValueError as e:
        logger.error(e)

    ####################################
    ## (META) CLUSTERING
    ####################################

    epsilon = config["epsilon"]
    clustering_name = config["clustering"]
    clusterer = model_factory_mapping[clustering_name]()
    clusterer.eps = epsilon
    clusterer.quantile = epsilon
    clusterer.n_clusters = None

    logger.info('grouping clustering data via {}'.format(clustering_name))

    random_state += 1
    list_cluster_IDs = clusterer.run(-1, npa_clusterings_reduced, random_state)

    list_df_metaclustered_elements = prepare_clustered_data(df_org_element2clusters, df_org_elements, list_cluster_IDs, config_UL, False)

    ####################################
    ## SOTRE & EXPORT RESULTS
    ####################################

    result = ExperimentResult(
        run_id=  0,
        n_dim=target_dimensionality,
        n_clusters=-1,
        transforming=transformer,
        clustering=clusterer,
        evaluation=None,
        df_elements_org=df_org_elements,
        trans_data=npa_clusterings_reduced,
        clustered_data=list_df_metaclustered_elements,
        cluster_IDs=list_cluster_IDs
    )

    # export results
    date_string = datetime.datetime.today().strftime("%y-%m-%d")
    time_string = datetime.datetime.now().strftime("%H-%M-%S")

    experiment_report_path = os.path \
        .join('reports/unsupervised/meta_clustering', date_string, time_string)
    os.makedirs(experiment_report_path, exist_ok=True)

    logger.info('exporting results to {}'.format(experiment_report_path))

    # write clusters to csv
    result.write_clusters_to_csv(experiment_report_path)

    # plot scatter charts &
    result.plot_results(experiment_report_path, config_UL)



