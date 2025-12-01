import matplotlib.pyplot as plt

def plot_dec_performance(missingness_percentages, score_arrays, labels, title):
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for i, scores in enumerate(score_arrays):
        color_idx = i % len(colors)
        marker_idx = i % len(markers)

        plt.plot(
            missingness_percentages,
            scores,
            label=labels[i],
            marker=markers[marker_idx],
            linestyle='-',
            linewidth=2,
            color=colors[color_idx]
        )

    plt.title(title, fontsize=14)
    plt.xlabel('MCAR Missingness Percentage', fontsize=12)
    plt.ylabel('Clustering Score', fontsize=12)

    plt.xticks(missingness_percentages)
    plt.legend(loc='best')

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()