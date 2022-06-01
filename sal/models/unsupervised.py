import logging
import os
import yaml
import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, axes

from sal.data.load import typed_view
from sal.features.scaling import z_score
from sal.features.selection import drop_labels
from sal.models import model_factory_mapping

import sal.visualization.sub_plotting as subplot
from sal.visualization.unsupervised import *


class ExperimentResult:
    run_id = None
    n_dim = None
    n_clusters = None

    transforming = None
    clustering = None
    evaluation = None

    trans_data = None
    clustered_data = None
    cluster_IDs = None

    def __init__(self, run_id, n_dim, n_clusters, transforming, clustering, evaluation, df_elements_org, trans_data, clustered_data, cluster_IDs):
        self.run_id = run_id
        self.n_dim = n_dim
        self.n_clusters = n_clusters
        self.transforming = transforming
        self.clustering = clustering
        self.evaluation = evaluation
        self.df_elements_org = df_elements_org
        self.trans_data = trans_data
        self.clustered_data = clustered_data
        self.cluster_IDs = cluster_IDs


    def get_name(self):
        name = \
            str(self.run_id) + " " + \
            self.transforming.name + "(#Dim=" + str(self.n_dim) + ") " + \
            self.clustering.name + "(#Cl=" + str(len(self.clustered_data)) + ") "

        if self.evaluation != None:
            name += "\n" + \
            self.evaluation.name
        return name

    def write_clustering_overview_to_file(self, path, compare_targets):
        file_object = open(path, "w")
        file_object.write(self.get_name() + "\n")
        file_object.write("Overall Cluster Difference: " + str(self.evaluation.overall_cluster_diff) + "\n\n")
        index = 0
        for cluster in self.clustered_data:
            file_object.write("Cluster " + str(index) + "\n" + "------------------" + "\n")
            for target in compare_targets:
                file_object.write(str(cluster[target].describe()) + "\n\n")
            file_object.write("==========================" + "\n")
            index += 1
        file_object.close()

    def write_clusters_to_csv(self, base_path):
        # extract clusters with original data
        clustered_org_data = get_cluster_groups(self.df_elements_org, self.cluster_IDs)

        index = 0
        for cluster in clustered_org_data:
            file_name = os.path.join(base_path, "cluster_" + str(index) + ".csv")
            cluster.to_csv(file_name, sep=";")

            index += 1

    def plot_results(self, path, config):
        name = self.get_name()

        if self.n_dim > 1:
            plotter_dimtrans = subplot.SubPlotter(6, 4, self.n_dim - 1, self.n_dim - 1, 0.1)
            plotter_dimtrans.plot_scatter_transposed_data(self.trans_data, self.n_dim, -1, 0)
            plotter_dimtrans.run(name, os.path.join(path, "transposition.jpg"))

            plotter_clustering = subplot.SubPlotter(6, 4, self.n_dim - 1, self.n_dim - 1, 0.1)
            plotter_clustering.plot_scatter_clustered_data(self.trans_data, self.cluster_IDs, self.n_dim, -1, 0)
            plotter_clustering.run(name, os.path.join(path, "clustering.jpg"))

        if self.evaluation != None:
            feature_count = len(self.evaluation.feature_diff_vectors[0][1])
            histo_plot_width = feature_count * 0.2
            histo_plot_heigth = histo_plot_width * 0.3

            plotter_featcomp = subplot.SubPlotter(histo_plot_width, histo_plot_heigth, len(self.clustered_data) - 1, len(self.clustered_data) - 1, 0.1)
            plotter_featcomp.plot_histo_feature_diff_vectors(self.evaluation.feature_diff_vectors, -1, 0, config['target_features'], config['visualization_min_single_cluster_diff'])
            plotter_featcomp.run(name, os.path.join(path, "cluster_feature_compare.jpg"))

        plot_clustering_overview(
            self.clustered_data,
            config['target_features'],
            path,
            "cluster_target_features.jpg")

        if self.n_dim == 3:
            plot_3D_mapping(self.trans_data, self.cluster_IDs, t_name=self.transforming.name, c_name=self.clustering.name)


def start_UL_experiments(input_file_path: str, config_file_path: str):
    logger = logging.getLogger(__name__)

    with open(config_file_path) as f:
        logger.info('config loaded from "{}"'.format(config_file_path))
        config = yaml.load(f, Loader=yaml.FullLoader)

        for experiment_set_name in config:
            if config[experiment_set_name] is not None:
                perform_UL_experiment_set(
                    experiment_set_name, input_file_path, config[experiment_set_name])

def get_cluster_groups(data_items, cluster_ids):
    found_clusters = {}
    for index in range(0, cluster_ids.shape[0]):
        cluster_id = cluster_ids[index]

        current_cluster = None
        if cluster_id in found_clusters:
            current_cluster = found_clusters[cluster_id]
        else:
            current_cluster = []
            found_clusters[cluster_id] = current_cluster

        current_item = data_items.iloc[index]
        current_cluster.append(current_item)

    cluster_list = []
    for key in found_clusters:
        cluster = pd.DataFrame(found_clusters[key])
        cluster_list.append(cluster)

    return cluster_list


def prepare_clustered_data(data, data_org, cluster_IDs, config, sanity_check=False):
    # (re-add compare labels for evaluation)
    data_with_objectives = data_org.copy()
    drop_labels(data_with_objectives, more=config['drop'])
    for compare_feature in config["target_features"]:
        data_with_objectives[compare_feature] = data_org[compare_feature]
    data_with_objectives = z_score(data_with_objectives)

    # split data regarding clustering
    clustered_org_data = get_cluster_groups(data_with_objectives, cluster_IDs)

    if sanity_check == False:
        return clustered_org_data

    # sanity check
    # print("TS: " + str(len(data)))
    for cluster in clustered_org_data:
        rel_size = float(len(cluster)) / float(len(data))
        #print("RL: " + str(rel_size))
        if rel_size < config['min_rel_cluster_size']:
            logging.info("abort evaluation - relative cluster size < " + str(config["min_rel_cluster_size"]))
            return None

    if len(clustered_org_data) > config["max_number_of_clusters"]:
        logging.info("abort evaluation - number of cluster > " + str(config["max_number_of_clusters"]))
        return None

    if len(clustered_org_data) < config["min_number_of_clusters"]:
        logging.info("abort evaluation - number of cluster < " + str(config["min_number_of_clusters"]))
        return None

    return clustered_org_data


def perform_clustering_and_evaluation(
        transformer, clusterer,
        run_id, target_number_of_dims, target_number_of_clusters,
        df, df_org, npa_trans_data,
        ExperimentList,
        config, random_state):

    # clustering of the transformed data
    cluster_ids = clusterer.run(target_number_of_clusters, npa_trans_data, random_state)

    # create seperate cluster lists & add compare labels (which were skipped previously to not distrub the transformation and clustering)
    clustered_org_df = prepare_clustered_data(df, df_org, cluster_ids, config, sanity_check = True)

    # sometimes an invalid cluster occurs (too few or too many clusters)
    if clustered_org_df is None:
        return False

    # measure clustering quality
    for evaluation_name in config['cluster_evaluation']:
        logging.info("starting experiments for cluster evaluation = " + str(evaluation_name))

        evaluator = model_factory_mapping[evaluation_name]()
        evaluator.initialize(evaluation_name, config)
        evaluator.run(clustered_org_df, npa_trans_data, cluster_ids)

        experiment_result = ExperimentResult(
            run_id,
            target_number_of_dims,
            target_number_of_clusters,
            transformer,
            clusterer,
            evaluator,
            df_org,
            npa_trans_data,
            clustered_org_df,
            cluster_ids)

        ExperimentList.append((experiment_result, evaluator.overall_cluster_diff))

        logging.info("done: experiments for cluster evaluation = " + str(evaluation_name))


def perform_UL_experiment_set(experiment_name, input_file_path, config: dict):
    logger = logging.getLogger(__name__)
    logger.info('conduct "{}" ...'.format(experiment_name))

    logger.info('data source: ' + input_file_path)

    logger.info('target features: ' + str(config["target_features"]) )
    logger.info('target dimensionality: ' + str(config["target_dimensionalities"]))
    logger.info('target number of clusters: ' + str(config["target_number_of_clusters"]))

    logger.info('feature drop:' + str(config['drop']))

    logger.info('transformation: ' + str(config['transformation']))
    logger.info('clustering: ' + str(config['clustering']))
    logger.info('cluster evaluation: ' + str(config['cluster_evaluation']))

    ##################################################
    ## Fetch & Prepare Data
    ##################################################

    # load original data only once, do not change this!
    df_org = typed_view(input_file_path)
    # working data, will be copied for each run
    df = df_org.copy()

    # filter & scale data
    drop_labels(df, more=config['drop'])
    df = z_score(df)

    df_save = df.copy()

    ##################################################
    ## Run Experiments
    ##################################################

    best_experiment = None
    ExperimentList = []

    random_state = config['random_seed']

    for run_id in range(0, config['runs_per_config']):
        logger.info("starting experiments for #run = " + str(run_id))

        for transformation_name in config['transformation']:
            logger.info("starting experiments for transformation = " + str(transformation_name))

            for target_number_of_dims in config['target_dimensionalities']:
                logger.info("starting experiments for #dimensionality = " + str(target_number_of_dims))

                # create transformation algorithm
                transformer = model_factory_mapping[transformation_name]()

                # data transformation
                # the catches are required because of some reported bug in the numpy svd module
                try:
                    random_state += 1
                    npa_trans_data = transformer.run(target_number_of_dims, df, random_state)
                except np.linalg.LinAlgError as e:
                    logger.error(e)
                    continue
                except ValueError as e:
                    logger.error(e)
                    continue


                # sometimes transformers are not able to return a valid result
                if npa_trans_data is None:
                    continue

                for clustering_name in config['clustering']:
                    logger.info("starting experiments for clustering = " + str(clustering_name))

                    # get clustering algorithm
                    clusterer = model_factory_mapping[clustering_name]()

                    random_state += 1

                    # differ between cluster algorithms, which need a specific number of cluster and those, which don't
                    if clusterer.requires_number_of_clusters() == True:
                        for target_number_of_clusters in config['target_number_of_clusters']:
                            logger.info("starting experiments for #clusters = " + str(target_number_of_clusters))
                            perform_clustering_and_evaluation(
                                transformer, clusterer,
                                run_id, target_number_of_dims, target_number_of_clusters,
                                df, df_org, npa_trans_data,
                                ExperimentList,
                                config, random_state)
                            logger.info("done: experiments for #clusters = " + str(target_number_of_clusters))
                    else:
                        perform_clustering_and_evaluation(
                            transformer, clusterer,
                            run_id, target_number_of_dims, -1,
                            df, df_org, npa_trans_data,
                            ExperimentList,
                            config, random_state)

                    logger.info("done: experiments for clustering = " + str(clustering_name))
                logger.info("done: experiments for #dimensionality = " + str(target_number_of_dims))
            logger.info("done: experiments for transformation = " + str(transformation_name))
        logger.info("done: experiments for #run = " + str(run_id))

    if len(ExperimentList) < 1:
        return False

    # sort the experiments w.r.t. overall cluster similarity
    ExperimentList_sorted = sorted(ExperimentList, key=lambda cluster_diff: cluster_diff[1])
    ExperimentList_sorted.reverse()

    # create base strings for result folders
    date_string = datetime.datetime.today().strftime("%y-%m-%d")
    time_string = datetime.datetime.now().strftime("%H-%M-%S")

    # create base folder
    experiment_report_base_path = os.path \
        .join('reports/unsupervised', date_string, time_string)
    os.makedirs(experiment_report_base_path, exist_ok=True)

    logger.info("writing max #results: " + str(config['max_number_of_results']))

    # store quality data for csv export
    df_qualities = pd.DataFrame(columns=[
        'experiment_name',
        'cluster_diff_metric',
        'silouette_coefficent',
        'calinski_harabasz_index',
        'davies_bouldin_score'
    ])

    df_cluster_assignments = pd.DataFrame(index=df_org.index)

    # write detailed information for the top X experiments
    index_detailed_result = 0
    index_detailed_clustering = 0

    for exp_result in ExperimentList_sorted:

        df_qualities = df_qualities.append({
            'experiment_name': exp_result[0].get_name(),
            'cluster_diff_metric': exp_result[1],
            'silouette_coefficent': exp_result[0].evaluation.silhouette_coef,
            'calinski_harabasz_index': exp_result[0].evaluation.calinski_harabasz_index,
            'davies_bouldin_score': exp_result[0].evaluation.davies_bouldin_score
        }, ignore_index=True)

        if config["export_detailed_cluster_assignments"]:
            df_cluster_assignments[str(index_detailed_clustering)] = exp_result[0].cluster_IDs
            index_detailed_clustering += 1

        if index_detailed_result < config['max_number_of_results']:
            current_path = os.path.join(experiment_report_base_path, str(index_detailed_result))
            os.makedirs(current_path, exist_ok=True)

            logger.info("writing results to: " + current_path)

            # write overall infos
            exp_result[0].write_clustering_overview_to_file(os.path.join(current_path, "clustering_overview.txt"), config['target_features'])

            # write clusters to csv
            exp_result[0].write_clusters_to_csv(current_path)

            # plot scatter charts &
            exp_result[0].plot_results(current_path, config)

            index_detailed_result += 1

    df_qualities.to_csv(os.path.join(experiment_report_base_path, "experiment_comparison.csv"), sep=';')

    if config["export_detailed_cluster_assignments"]:
        df_cluster_assignments.to_csv(os.path.join(experiment_report_base_path, "element_cluster_assignments.csv"), sep=';')

    plot_experiments_overview(
        ExperimentList_sorted,
        25,
        experiment_report_base_path,
        "experiment_comparison.jpg")

    #plt.show()



