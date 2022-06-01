#region imports
from matplotlib import pyplot as plt
#endregion

#region plotters
class SubPlotter:
    base_fig_width = 0
    base_fig_height = 0
    fig = None
    axs = None
    scatter_alpha = 1

    def plot_scatter_transposed_data(self, data, dim_count, column_offset = 0, row_offset = 0):
        for d1 in range(0, dim_count - 1):
            for d2 in range(d1 + 1, dim_count - 0):
                current_axis = self.axs[d1 + row_offset, d2 + column_offset]
                current_axis.scatter(data[:, d1], data[:, d2], alpha=self.scatter_alpha)
                current_axis.set_title("Dimension " + str(d1) + "-" + str(d2))

    def plot_scatter_clustered_data(self, data, cluster_ids, dim_count, column_offset = 0, row_offset = 0):
        for d1 in range(0, dim_count - 1):
            for d2 in range(d1 + 1, dim_count - 0):
                current_axis = self.axs[d1 + row_offset, d2 + column_offset]
                current_axis.scatter(data[:, d1], data[:, d2], c=cluster_ids, alpha=self.scatter_alpha)
                current_axis.set_title("Dimension " + str(d1) + "-" + str(d2))

    def plot_histo_feature_diff_vectors(self, diff_vectors, column_offset, row_offset, target_names, min_single_cluster_diff):
        for y in diff_vectors:
            for x in diff_vectors[y]:
                diff_vector = diff_vectors[y][x]
                title = "Difference Clusters " + str(y) + "-" + str(x)
                self.plot_histo_feature_diff(diff_vector, title, x + column_offset, y + row_offset, target_names, min_single_cluster_diff)

    def plot_histo_feature_diff(self, feat_diff_vector, title, column_index, row_index, target_names, min_single_cluster_diff):
        current_axis = self.axs[row_index, column_index]

        # set colors & filter entries
        filtered_feat_diff_vector = []
        for index, tuple in enumerate(feat_diff_vector):
            feat_name = tuple[0]
            feat_diff_value = tuple[1]
            if feat_name in target_names or feat_diff_value > min_single_cluster_diff:
                filtered_feat_diff_vector.append( (feat_name, feat_diff_value) )

        barcolors = []
        for index, tuple in enumerate(filtered_feat_diff_vector):
            feat_name = tuple[0]
            if feat_name in target_names:
                barcolors.append('r')
            else:
                barcolors.append('b')

        Xs = (range(len(filtered_feat_diff_vector)))

        current_axis.bar(Xs, [val[1] for val in filtered_feat_diff_vector], align='center', width=0.8, color = barcolors)
        current_axis.set_xticks(Xs)
        current_axis.set_xticklabels([val[0] for val in filtered_feat_diff_vector], rotation=90)
        for tick in current_axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(7)
        current_axis.set_title(title)
        current_axis.grid()


    def __init__(self, base_fig_width, base_fig_height, column_count, row_count, scatter_alpha = 0.3):
        self.base_fig_width = base_fig_width
        self.base_fig_height = base_fig_height
        self.scatter_alpha = scatter_alpha

        self.fig, self.axs = plt.subplots(row_count, column_count, squeeze=False)
        #self.fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        self.fig.set_figwidth(base_fig_width * column_count)
        self.fig.set_figheight(base_fig_height * row_count)

    def run(self, title, filename = None):
        self.fig.suptitle(title, fontsize=16, y=1.00)
        self.fig.subplots_adjust(hspace=1.0)

        self.fig.tight_layout()

        if filename is not None:
            self.fig.savefig(filename, dpi=300)

        plt.close()
        #self.fig.show()
#endregion