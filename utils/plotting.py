import matplotlib.pyplot as plt
import torch


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


def plot_experiment_results(
        missing_rates,
        ari_scores_mean, ari_scores_knn, ari_scores_dae,
        nmi_scores_mean, nmi_scores_knn, nmi_scores_dae
):
    plt.figure(figsize=(12, 6))

    # ---------------- ARI ---------------- #
    plt.subplot(1, 2, 1)
    plt.plot(missing_rates, ari_scores_mean, label="Mean Imputer", marker='o')
    plt.plot(missing_rates, ari_scores_knn, label="kNN Imputer", marker='o')
    plt.plot(missing_rates, ari_scores_dae, label="DAE", marker='o')

    plt.title("ARI vs Missing Rate", fontsize=14)
    plt.xlabel("Missing Rate (%)", fontsize=12)
    plt.ylabel("ARI", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # ---------------- NMI ---------------- #
    plt.subplot(1, 2, 2)
    plt.plot(missing_rates, nmi_scores_mean, label="Mean Imputer", marker='o')
    plt.plot(missing_rates, nmi_scores_knn, label="kNN Imputer", marker='o')
    plt.plot(missing_rates, nmi_scores_dae, label="DAE", marker='o')

    plt.title("NMI vs Missing Rate", fontsize=14)
    plt.xlabel("Missing Rate (%)", fontsize=12)
    plt.ylabel("NMI", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


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
