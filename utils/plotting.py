import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix

from models.gmm import GMMMissing


def plot_dec_performance(
        missingness_percentages,
        score_arrays,
        labels,
        title
):
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


def plot_performance_average(
        missingness_percentages,
        score_arrays,
        labels,
        title='Clustering Performance (All Metrics)'
):
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(10, 6))

    for i, (runs, label) in enumerate(zip(score_arrays, labels)):
        runs = np.array(runs)
        mean_curve = runs.mean(axis=0)
        std_curve = runs.std(axis=0)

        plt.fill_between(
            missingness_percentages,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=colors[i]
        )

        plt.plot(
            missingness_percentages,
            mean_curve,
            label=f"{label} (mean ± std)",
            color=colors[i],
            marker='o',
            linewidth=2
        )

    # max_val = missingness_percentages.max()
    # step = 5 if max_val <= 30 else 10
    #
    # ticks = np.arange(0, max_val + 1e-9, step)
    # plt.xticks(ticks)

    plt.title(title, fontsize=16)
    plt.xlabel("Missingness Level", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    # plt.xticks(missingness_percentages)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_dec_experiment_results(
        missing_rates,
        ari_scores_mean, ari_scores_knn, ari_scores_dae,
        nmi_scores_mean, nmi_scores_knn, nmi_scores_dae,
        acc_scores_mean, acc_scores_knn, acc_scores_dae
):
    plt.figure(figsize=(26, 6))

    # ARI
    plt.subplot(1, 3, 1)
    plt.plot(missing_rates, ari_scores_mean, label="Mean Imputer", marker='o')
    plt.plot(missing_rates, ari_scores_knn, label="kNN Imputer", marker='o')
    plt.plot(missing_rates, ari_scores_dae, label="DAE", marker='o')

    plt.title("ARI vs Missing Rate", fontsize=14)
    plt.xlabel("Missing Rate (%)", fontsize=12)
    plt.ylabel("ARI", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(list(range(0, 100, 10)))
    plt.legend()

    # NMI
    plt.subplot(1, 3, 2)
    plt.plot(missing_rates, nmi_scores_mean, label="Mean Imputer", marker='o')
    plt.plot(missing_rates, nmi_scores_knn, label="kNN Imputer", marker='o')
    plt.plot(missing_rates, nmi_scores_dae, label="DAE", marker='o')

    plt.title("NMI vs Missing Rate", fontsize=14)
    plt.xlabel("Missing Rate (%)", fontsize=12)
    plt.ylabel("NMI", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(list(range(0, 100, 10)))
    plt.legend()

    # ACC
    plt.subplot(1, 3, 3)
    plt.plot(missing_rates, acc_scores_mean, label="Mean Imputer", marker='o')
    plt.plot(missing_rates, acc_scores_knn, label="kNN Imputer", marker='o')
    plt.plot(missing_rates, acc_scores_dae, label="DAE", marker='o')

    plt.title("Accuracy vs Missing Rate", fontsize=14)
    plt.xlabel("Missing Rate (%)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(list(range(0, 100, 10)))
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_experiment_results_average(
        missingness_levels,
        ari_runs_mean, ari_runs_knn, ari_runs_dae,
        nmi_runs_mean, nmi_runs_knn, nmi_runs_dae,
        acc_runs_mean, acc_runs_knn, acc_runs_dae,
        title="DEC Performance Across Imputation Methods"
):
    plt.figure(figsize=(26, 6))

    method_colors = {
        "Mean": "blue",
        "kNN": "green",
        "DAE": "red"
    }

    # ARI
    plt.subplot(1, 3, 1)
    _plot_metric_mean_std(
        missingness_levels,
        [ari_runs_mean, ari_runs_knn, ari_runs_dae],
        ["Mean", "kNN", "DAE"],
        [method_colors["Mean"], method_colors["kNN"], method_colors["DAE"]],
        "ARI"
    )

    # NMI
    plt.subplot(1, 3, 2)
    _plot_metric_mean_std(
        missingness_levels,
        [nmi_runs_mean, nmi_runs_knn, nmi_runs_dae],
        ["Mean", "kNN", "DAE"],
        [method_colors["Mean"], method_colors["kNN"], method_colors["DAE"]],
        "NMI"
    )

    # ACC
    plt.subplot(1, 3, 3)
    _plot_metric_mean_std(
        missingness_levels,
        [acc_runs_mean, acc_runs_knn, acc_runs_dae],
        ["Mean", "kNN", "DAE"],
        [method_colors["Mean"], method_colors["kNN"], method_colors["DAE"]],
        "Accuracy"
    )

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_experiment_results(
        missingness_levels,
        results,
        title="Clustering Performance Across Missing-Data Strategies"
):
    plt.figure(figsize=(6 * len(results), 6))

    metrics = ["ARI", "NMI", "ACC"]
    colors = plt.cm.tab10.colors

    for i, metric in enumerate(metrics, start=1):
        plt.subplot(1, 3, i)

        for (method, method_results), color in zip(results.items(), colors):
            _plot_metric_mean_std(
                missingness_levels,
                [method_results[metric]],
                [method],
                [color],
                metric
            )

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def _plot_metric_mean_std(x_values, metric_runs, labels, colors, metric_name):
    for runs, label, color in zip(metric_runs, labels, colors):
        runs = np.array(runs)
        mean_curve = runs.mean(axis=0)
        std_curve = runs.std(axis=0)

        plt.fill_between(
            x_values,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.25,
            color=color
        )

        plt.plot(
            x_values,
            mean_curve,
            color=color,
            marker="o",
            linewidth=2,
            label=f"{label} (mean ± std)"
        )

    plt.title(metric_name, fontsize=16)
    plt.xlabel("Missingness (%)", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    # plt.xticks(x_values)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()


def plot_ae_reconstructions(
        model,
        dataset,
        device='cpu',
        missingness=None,
        title=None,
        corruption_type="mcar",
        **corruption_kwargs,

):
    model.eval()

    indices = get_one_index_per_label(dataset)
    n = len(indices)

    originals = []
    corrupteds = []
    recons = []

    cmap = plt.cm.gray.copy()
    cmap.set_bad(color="red")

    with torch.no_grad():
        for idx in indices:
            img, _ = dataset[idx]
            originals.append(img.squeeze())

            corrupted, mask = missingness.apply_corruption(
                img.unsqueeze(0),
                corruption_type,
                **corruption_kwargs
            )
            corrupted_display = corrupted.squeeze()
            corrupted_for_model = torch.nan_to_num(corrupted, nan=0.0)

            corrupteds.append(corrupted_display.cpu())

            x_hat, _ = model(corrupted_for_model.view(1, -1).to(device))

            if x_hat.dim() == 2:
                x_hat = x_hat.view(1, 1, 28, 28)

            recons.append(x_hat.squeeze().cpu())

    rows = 3
    cols = n
    plt.figure(figsize=(2.2 * n, 6))

    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(originals[i], cmap=cmap)
        ax.set_title("Original")
        plt.axis("off")

    for i in range(n):
        ax = plt.subplot(rows, cols, n + i + 1)
        plt.imshow(corrupteds[i], cmap=cmap)
        ax.set_title("Corrupted")
        plt.axis("off")

    for i in range(n):
        ax = plt.subplot(rows, cols, 2 * n + i + 1)
        plt.imshow(recons[i], cmap=cmap)
        ax.set_title("Reconstructed")
        plt.axis("off")

    if title is not None:
        plt.suptitle(title, fontsize=20)

    plt.tight_layout()
    plt.show()


def plot_all_reconstructions(images_dict):
    n_images = len(images_dict)
    n_recons = max(len(v["reconstructions"]) for v in images_dict.values())

    fig, axes = plt.subplots(
        n_images,
        n_recons + 1,
        figsize=(3 * (n_recons + 1), 3 * n_images)
    )

    if n_images == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (img_index, data) in enumerate(images_dict.items()):

        # Original image
        original = data["original"].detach().cpu().squeeze()
        original = original.view(28, 28)
        axes[row_idx, 0].imshow(original, cmap="gray")
        axes[row_idx, 0].set_title(f"Original (idx={img_index})")
        axes[row_idx, 0].axis("off")

        # All the reconstructions
        for col_idx in range(n_recons):
            ax = axes[row_idx, col_idx + 1]

            if col_idx < len(data["reconstructions"]):
                x_hat = data["reconstructions"][col_idx]

                if x_hat.dim() == 2:
                    img = x_hat.view(28, 28)
                else:
                    img = x_hat.squeeze()

                img = img.detach().cpu()

                ax.imshow(img, cmap="gray")
                ax.set_title(f"Epoch {col_idx + 1}")
            else:
                ax.axis("off")

            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_ae_losses(ae_losses, title=None):
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(ae_losses) + 1),
        ae_losses,
        linewidth=2,
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title(title, fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def get_one_index_per_label(dataset):
    indices = [-1] * 10

    for idx in range(len(dataset)):
        _, label = dataset[idx]

        if indices[label] == -1:
            indices[label] = idx

        if all(i != -1 for i in indices):
            break

    return indices


def plot_imputation_and_alignment(X, y, n_classes, missing):
    gmm = GMMMissing(n_components=n_classes, random_state=42)
    gmm.fit(missing)

    prediction = gmm.predict(missing)
    imputed = gmm.impute(missing, stochastic=True)

    cm = contingency_matrix(y, prediction)
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_predictions = np.array([mapping[p] for p in prediction])
    errors = aligned_predictions != y

    is_imputed = np.isnan(missing).any(axis=1)

    use_pca = X.shape[1] > 2

    if use_pca:
        pca = PCA(n_components=2, random_state=42)
        X_plot = pca.fit_transform(X)
        imputed_plot = pca.transform(imputed)
        x_label, y_label = "PC 1", "PC 2"
    else:
        X_plot = X
        imputed_plot = imputed
        x_label, y_label = "Feature 0", "Feature 1"

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0, 0].scatter(X_plot[:, 0], X_plot[:, 1], c=y, s=30, alpha=0.7, cmap="viridis")
    ax[0, 0].set_title("Original Data (Ground Truth)")
    ax[0, 1].scatter(imputed_plot[:, 0], imputed_plot[:, 1], c=is_imputed, s=30, alpha=0.6, cmap="autumn")
    ax[0, 1].set_title("Imputed Data (Yellow = Filled)")
    ax[1, 0].scatter(imputed_plot[:, 0], imputed_plot[:, 1], c=aligned_predictions, s=30, alpha=0.7, cmap="viridis")
    ax[1, 0].set_title("Aligned Predictions (Mapped to True Class)")
    ax[1, 1].scatter(imputed_plot[:, 0], imputed_plot[:, 1], c=errors, s=30, alpha=0.7, cmap="coolwarm")
    ax[1, 1].set_title("Clustering Errors (Red = Misclassified)")

    for a in ax.flat:
        a.set_xlabel(x_label)
        a.set_ylabel(y_label)
        a.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.show()


def plot_latent_space(model, tensor_x, y_true, device='cpu'):
    model.eval()

    with torch.no_grad():
        _, z = model(tensor_x.to(device))
        z = z.cpu().numpy()

    if z.shape[1] > 2:
        pca = PCA(n_components=2)
        z_plot = pca.fit_transform(z)
        print('Using PCA')
    else:
        z_plot = z

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        z_plot[:, 0], z_plot[:, 1],
        c=y_true, cmap='viridis',
        s=45, alpha=0.8, edgecolors='white', linewidth=0.5
    )

    plt.colorbar(scatter, label='Class Label')
    plt.title(f"Autoencoder: Latent Space Representation", fontsize=13, pad=15)
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_comparison(model, tensor_x, y_true, device='cpu'):
    model.eval()
    with torch.no_grad():
        x_hat, _ = model(tensor_x.to(device))
        x_hat = x_hat.cpu().numpy()

    X_orig = tensor_x.cpu().numpy()

    pca = PCA(n_components=2)

    z_orig = pca.fit_transform(X_orig)
    z_hat = pca.fit_transform(x_hat)

    z_orig = X_orig
    z_hat = x_hat

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    scatter1 = axes[0].scatter(z_orig[:, 0], z_orig[:, 1], c=y_true, cmap='viridis', s=40, alpha=0.7)
    axes[0].set_title("Original Data (PCA Projection)", fontsize=14)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    scatter2 = axes[1].scatter(z_hat[:, 0], z_hat[:, 1], c=y_true, cmap='viridis', s=40, alpha=0.7)
    axes[1].set_title("AE Reconstructed Data (PCA Projection)", fontsize=14)
    axes[1].set_xlabel("PC1")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    cbar = fig.colorbar(scatter1, ax=axes, orientation='vertical', fraction=.02, pad=0.04)
    cbar.set_label('Species Class')

    plt.suptitle("Original vs Reconstructed", fontsize=16)
    plt.show()
