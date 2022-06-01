#region imports
from pandas.api.types import is_numeric_dtype
from statistics import median, mean

from sklearn import metrics
from sklearn.metrics import pairwise_distances
#endregion


#region base class
class SALA_ClusterEvaluator:
    name = ""
    experiment_title = ""
    console_output = False

    overall_cluster_diff = -1
    silhouette_coef = -1
    calinski_harabasz_index = -1
    davies_bouldin_score = -1

    feature_diff_vectors = {}

    def __init__(self, name = "not set", config=None):
        self.name = name

    # base feature comparisons
    def CalcAbsDiff(self, feat_value_1, feat_value_2):
        return abs(feat_value_1 - feat_value_2)

    def CalcRelDiff(self, feat_value_1, feat_value_2):
        return max(feat_value_1, feat_value_2) / min(feat_value_1, feat_value_2)



    # external cluster quality evaluation
    def calc_silhouette_coef(self, data, cluster_ids):
        result = metrics.silhouette_score(data, cluster_ids, metric='euclidean')
        return result

    def calc_calinski_harabasz_index(self, data, cluster_ids):
        result = metrics.calinski_harabasz_score(data, cluster_ids)
        return result

    def calc_davies_bouldin_score(self, data, cluster_ids):
        result = metrics.davies_bouldin_score(data, cluster_ids)
        return result


    # main run method
    def run(self, clusters, transposed_data, cluster_ids):
        if self.console_output:
            print("Start " + self.name)

        count = 0
        cluster_diff_sum = 0

        self.feature_diff_vectors = {}

        for i in range(0, len(clusters)-1):
            self.feature_diff_vectors[i] = {}

            for j in range(i+1, len(clusters)):
                if self.console_output:
                    print("  Clusters: " + str(i) + ", " + str(j))

                cluster_diff_vector = self.CalcClusterDiffVector(clusters[i], clusters[j])
                self.feature_diff_vectors[i][j] = cluster_diff_vector

                #if self.show_histo_plot:
                #    self.histo_plot(cluster_diff_vector, self.name + " " + str(i) + "-" + str(j))

                cluster_diff = self.CalcClusterDiffValue(cluster_diff_vector)

                cluster_weight_sum = len(clusters[i]) + len(clusters[j])
                cluster_diff_sum += cluster_diff * cluster_weight_sum
                count += cluster_weight_sum

                if self.console_output:
                    print("  Cluster Diff: " + str(cluster_diff))

        self.overall_cluster_diff = cluster_diff_sum / count

        self.silhouette_coef = self.calc_silhouette_coef(transposed_data, cluster_ids)
        self.calinski_harabasz_index = self.calc_calinski_harabasz_index(transposed_data, cluster_ids)
        self.davies_bouldin_score = self.calc_davies_bouldin_score(transposed_data, cluster_ids)

        if self.console_output:
            print("  Overall Cluster Diff: " + str(self.overall_cluster_diff))
            print("End " + self.name)

        return self.overall_cluster_diff

    def CalcClusterDiffVector(self, cluster_1, cluster_2):
        column_names = cluster_1.columns.copy()
        cluster_diffs_by_metric = []

        for column_name in column_names:
            feat_1 = cluster_1[column_name]
            feat_2 = cluster_2[column_name]

            if is_numeric_dtype(feat_1) and is_numeric_dtype(feat_2):
                diff_specific = self.CalcFeatValueDiff(feat_1, feat_2)
                cluster_diffs_by_metric.append((column_name, diff_specific))

        sorted_cluster_diffs = sorted(cluster_diffs_by_metric, key=lambda mean_diff: mean_diff[1])
        sorted_cluster_diffs.reverse()

        if self.console_output:
            print("  " + str(sorted_cluster_diffs))

        return sorted_cluster_diffs

    # scaling function to handle differences in various range (1 vs 2 can be considered more different than 1001 vs 1002)
    # returns a scaling factor for a given value
    def GlobalScaling(self, value):
        pass


    # returns a value for the comparison of two given features
    def CalcFeatDiff(self, feat_1, feat_2):
        pass

    # return a value for the comparion of two given clusters (feature lists), depends an CalcFeatDiff and GlobalScaling
    def CalcClusterDiffValue(self, feat_diff_vector):
        pass


#endregion


#region evaluators
class SALA_Saturated_Pairwise_Mean(SALA_ClusterEvaluator):
    def __init__(self, config=None):
        SALA_ClusterEvaluator.__init__(self, "SPMn", config)

    def GlobalScaling(self, value):
        if value >= 0:
            return 1 / (value + 1)
        else:
            return -1 / (value - 1)

    def CalcFeatValueDiff(self, feat_1, feat_2):
        mean_1 = feat_1.mean()
        mean_2 = feat_2.mean()

        center = (mean_1 + mean_2) / 2
        scaling = self.GlobalScaling(center)
        diff_abs = self.CalcAbsDiff(mean_1, mean_2)
        diff_specific = diff_abs * scaling
        return diff_specific

    def CalcClusterDiffValue(self, feat_diff_vector):
        value_list = []
        for index, tup in enumerate(feat_diff_vector):
            value_list.append(tup[1])

        result = mean(value_list)
        return result


class SALA_Pairwise_Median(SALA_ClusterEvaluator):
    def __init__(self, config=None):
        SALA_ClusterEvaluator.__init__(self, "PMd", config)

    def CalcFeatValueDiff(self, feat_1, feat_2):
        median_1 = feat_1.median()
        median_2 = feat_2.median()

        diff_abs = self.CalcAbsDiff(median_1, median_2)
        return diff_abs

    def CalcClusterDiffValue(self, feat_diff_vector):
        value_list = []
        for index, tup in enumerate(feat_diff_vector):
            value_list.append(tup[1])

        result = mean(value_list)
        return result


#endregion




class SALA_StagedClusterEvaluator:
    name = ""
    experiment_title = ""
    config = None
    specific_config = None

    console_output = False

    overall_cluster_diff = -1
    silhouette_coef = -1
    calinski_harabasz_index = -1
    davies_bouldin_score = -1

    feature_diff_vectors = {}

    def __init__(self, name="not set", config=None):
        self.name = "not set"

    def initialize(self, config_name, config):
        self.config = config
        self.specific_config = self.config["cluster_evaluation"][config_name]

        self.name = \
            "FA:" + self.specific_config["stage1"] + " " + \
            "FS:" + self.specific_config["stage2"] + " " + \
            "FC:" + self.specific_config["stage3"] + " " + \
            "CA:" + self.specific_config["stage4"]

    # base feature comparisons
    def CalcAbsDiff(self, feat_value_1, feat_value_2):
        return abs(feat_value_1 - feat_value_2)

    def CalcRelDiff(self, feat_value_1, feat_value_2):
        return max(abs(feat_value_1) + 1, abs(feat_value_2) + 1) / \
               min(abs(feat_value_1) + 1, abs(feat_value_2) + 1)
        #if min(feat_value_1, feat_value_2) != 0:
        #    return max(feat_value_1, feat_value_2) / min(feat_value_1, feat_value_2)
        #else:
        #    return 1

    # external cluster quality evaluation
    def calc_silhouette_coef(self, data, cluster_ids):
        result = metrics.silhouette_score(data, cluster_ids, metric='euclidean')
        return result

    def calc_calinski_harabasz_index(self, data, cluster_ids):
        result = metrics.calinski_harabasz_score(data, cluster_ids)
        return result

    def calc_davies_bouldin_score(self, data, cluster_ids):
        result = metrics.davies_bouldin_score(data, cluster_ids)
        return result




    def stage1_intra_cluster_feature_vector_aggregator(self, feature_vector):
        if self.specific_config["stage1"] == "Mean":
            return feature_vector.mean()

        if self.specific_config["stage1"] == "Median":
            return feature_vector.median()


    def stage2_intra_cluster_feature_diff_scaling(self, feature_diff, reference_value):
        scaling = None

        if self.specific_config["stage2"] == "Const":
            scaling = 1

        if self.specific_config["stage2"] == "Fade":
            if reference_value >= 0:
                scaling = 1 / (reference_value + 1)
            else:
                scaling = -1 / (reference_value - 1)

        feature_diff_scaled = feature_diff * scaling

        return feature_diff_scaled


    def stage3_inter_cluster_feature_comparator(self, feat_1, feat_2):
        feat_1_aggregated = self.stage1_intra_cluster_feature_vector_aggregator(feat_1)
        feat_2_aggregated = self.stage1_intra_cluster_feature_vector_aggregator(feat_2)

        feature_diff = None

        if self.specific_config["stage3"] == "AbsDiff":
            feature_diff = self.CalcAbsDiff(feat_1_aggregated, feat_2_aggregated)

        if self.specific_config["stage3"] == "RelDiff":
            feature_diff = self.CalcRelDiff(feat_1_aggregated, feat_2_aggregated)

        reference_value = (feat_1_aggregated + feat_2_aggregated) / 2
        feature_diff_scaled = self.stage2_intra_cluster_feature_diff_scaling(feature_diff, reference_value)

        return feature_diff_scaled



    def stage3_inter_cluster_feature_vector_comparator(self, cluster_1, cluster_2):
        column_names = cluster_1.columns.copy()
        cluster_diffs_by_metric = []

        for column_name in column_names:
            feat_1 = cluster_1[column_name]
            feat_2 = cluster_2[column_name]

            if is_numeric_dtype(feat_1) and is_numeric_dtype(feat_2):
                diff_specific = self.stage3_inter_cluster_feature_comparator(feat_1, feat_2)
                cluster_diffs_by_metric.append((column_name, diff_specific))

        sorted_cluster_diffs = sorted(cluster_diffs_by_metric, key=lambda mean_diff: mean_diff[1])
        sorted_cluster_diffs.reverse()

        if self.console_output:
            print("  " + str(sorted_cluster_diffs))

        return sorted_cluster_diffs

    def stage4_inter_cluster_difference_aggregation(self, feat_diff_vector):
        value_list = []
        for index, tup in enumerate(feat_diff_vector):
            value_list.append(tup[1])

        if self.specific_config["stage4"] == "Mean":
            result = mean(value_list)
            return result

        if self.specific_config["stage4"] == "Median":
            result = median(value_list)
            return result



    # main run method
    def run(self, clusters, transposed_data, cluster_ids):
        if self.console_output:
            print("Start " + self.name)

        count = 0
        cluster_diff_sum = 0

        self.feature_diff_vectors = {} # cluster index -> target cluster index -> vector of tubles (name, diff)

        for i in range(0, len(clusters) - 1):
            self.feature_diff_vectors[i] = {}

            for j in range(i + 1, len(clusters)):
                if self.console_output:
                    print("  Clusters: " + str(i) + ", " + str(j))

                cluster_diff_vector = self.stage3_inter_cluster_feature_vector_comparator(clusters[i], clusters[j])
                self.feature_diff_vectors[i][j] = cluster_diff_vector

                # if self.show_histo_plot:
                #    self.histo_plot(cluster_diff_vector, self.name + " " + str(i) + "-" + str(j))

                cluster_diff = self.stage4_inter_cluster_difference_aggregation(cluster_diff_vector)

                cluster_weight_sum = len(clusters[i]) + len(clusters[j])
                cluster_diff_sum += cluster_diff * cluster_weight_sum
                count += cluster_weight_sum

                if self.console_output:
                    print("  Cluster Diff: " + str(cluster_diff))

        self.overall_cluster_diff = cluster_diff_sum / count

        self.silhouette_coef = self.calc_silhouette_coef(transposed_data, cluster_ids)
        self.calinski_harabasz_index = self.calc_calinski_harabasz_index(transposed_data, cluster_ids)
        self.davies_bouldin_score = self.calc_davies_bouldin_score(transposed_data, cluster_ids)

        if self.console_output:
            print("  Overall Cluster Diff: " + str(self.overall_cluster_diff))
            print("End " + self.name)

        return self.overall_cluster_diff



    # scaling function to handle differences in various range (1 vs 2 can be considered more different than 1001 vs 1002)
    # returns a scaling factor for a given value
    def GlobalScaling(self, value):
        pass

    # returns a value for the comparison of two given features
    def CalcFeatDiff(self, feat_1, feat_2):
        pass

    # return a value for the comparion of two given clusters (feature lists), depends an CalcFeatDiff and GlobalScaling
    def CalcClusterDiffValue(self, feat_diff_vector):
        pass
