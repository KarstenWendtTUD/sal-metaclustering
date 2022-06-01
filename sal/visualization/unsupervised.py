from matplotlib import pyplot as plt, axes
import os
import numpy as np


def plot_experiments_overview(ExperimentList, max_x_label, experiment_report_base_path, name):
    names = []
    cluster_diff_metrics = []

    silouette_coeffs = []
    max_silouette_coeff = float("-inf")

    calinski_harabasz_indices = []
    max_calinski_harabasz_index = float("-inf")

    davies_bouldin_scores = []
    max_davies_bouldin_score = float("-inf")

    for exp_result in ExperimentList:
        experiment = exp_result[0]
        cluster_diff_metric = exp_result[1]

        names.append(experiment.get_name())
        cluster_diff_metrics.append(cluster_diff_metric)

        silouette_coeffs.append(experiment.evaluation.silhouette_coef)
        max_silouette_coeff = max(max_silouette_coeff, experiment.evaluation.silhouette_coef)

        calinski_harabasz_indices.append((experiment.evaluation.calinski_harabasz_index))
        max_calinski_harabasz_index = max(max_calinski_harabasz_index, experiment.evaluation.calinski_harabasz_index)

        davies_bouldin_scores.append((experiment.evaluation.davies_bouldin_score))
        max_davies_bouldin_score = max(max_davies_bouldin_score, experiment.evaluation.davies_bouldin_score)

    fig = plt.figure(figsize=(15, 10))
    plt.xticks(rotation=90, fontsize=10)
    plt.plot(names, cluster_diff_metrics,
             color="blue", label="Cluster Difference Metric", linewidth=2)

    ax = plt.axes()
    ax2 = ax.twinx()

    ax2.plot(names, silouette_coeffs / max_silouette_coeff,
             color="red", ls="--", lw=0.7,
             label="Rel. Silouette Coefficent")
    ax2.plot(names, calinski_harabasz_indices / max_calinski_harabasz_index,
             color="green", ls="--", lw=0.7,
             label="Rel. Calinski Harabasz_Index")
    ax2.plot(names, davies_bouldin_scores / max_davies_bouldin_score,
             color="orange", ls="--", lw=0.7,
             label="Rel. Davies-Bouldin Score")

    ax.grid()
    ax.legend(loc='lower left')
    ax2.legend(loc='upper right')
    #ax2.set_yscale('log')

    if len(ExperimentList) > max_x_label:
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_x_label))



    plt.title("Cluster Similarity & Quality Overview")
    plt.tight_layout()

    fig.savefig(os.path.join(experiment_report_base_path, name), dpi=300)


def plot_clustering_overview(clustered_data, target_features, experiment_report_base_path, name):
    cluster_count = len(clustered_data)
    x_indices = np.arange(cluster_count)
    barwidth = 0.8 / float(len(target_features))

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes()

    x_labels = []
    id = 0
    for cluster in clustered_data:
        x_labels.append("Cluster " + str(id) + " (" + str(cluster.shape[0]) + ")")
        id += 1

    legend_items = []
    legend_labels = []

    position = 0
    for target_feature_name in target_features:
        values = []
        errors = []
        for cluster in clustered_data:
            target_feature = cluster[target_feature_name]
            values.append(target_feature.mean())
            errors.append(target_feature.std())

        #rectangles = ax.bar(x_indices + barwidth * position, values, barwidth, yerr=errors)
        rectangles = ax.bar(x_indices + barwidth * position, values, barwidth)

        bar_index = 0
        for rect in rectangles:
            rect_heights = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0,
                    rect_heights + 0,
                    "mean: {:4.2f}".format(values[bar_index]) + "\n" + "std: {:4.2f}".format(errors[bar_index]),
                    fontsize=8,
                    ha='center', va='bottom')
            bar_index += 1

        legend_items.append(rectangles)
        legend_labels.append(target_feature_name)

        position += 1

    ax.set_ylabel('Mean Feature Value')
    ax.set_xticks(x_indices + barwidth)

    ax.set_xticklabels(x_labels)
    ax.legend(legend_items, legend_labels)
    plt.title("Target Feature per Cluster")

    #def autolabel(rects):
    #    for rect in rects:
    #        h = rect.get_height()
    #        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
    #                ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)

    plt.tight_layout()
    fig.savefig(os.path.join(experiment_report_base_path, name), dpi=300)

def plot_3D_mapping(trans_data, clustering, t_name, c_name):
    xs = []
    ys = []
    zs = []

    color_mapping = {
        0: [1.0, 0.0, 0.0],
        1: [0.0, 1.0, 0.0],
        2: [0.0, 0.0, 1.0],
        3: [1.0, 1.0, 0.0],
        4: [1.0, 0.0, 1.0],
        5: [0.0, 1.0, 0.0],
        6: [0.0, 0.0, 0.0]
    }
    alpha = 0.2

    colors = []

    for cluster_id in clustering:
        colors.append(color_mapping[cluster_id])

    for item in trans_data:
        xs.append(item[0])
        ys.append(item[1])
        zs.append(item[2])

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes()

    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(xs, ys, zs, c=colors, alpha=alpha, depthshade=True)

    #ax.plot(xs, ys, 'o', zdir='z', zs=0.0)

    #ax.set_xlim([-0.5, 1.5])
    #ax.set_ylim([-0.5, 1.5])
    #ax.set_zlim([-1.5, 1.5])

    ax.set_xlabel(t_name + ' - X')
    ax.set_ylabel(t_name + ' - Y')
    ax.set_zlabel(t_name + ' - Z')

    plt.title(t_name + " + " + c_name)

    plt.show()