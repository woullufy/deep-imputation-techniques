import matplotlib.pyplot as plt
import torch


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


def plot_ae_reconstructions(
        model,
        dataset,
        n=10,
        device=None,
        indices=None,
        missingness=None,
        corruption_type="mcar",
        **corruption_kwargs
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    if indices is None:
        indices = torch.randint(0, len(dataset), size=(n,))
    else:
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
    plt.figure(figsize=(2.5 * n, 6))

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

    plt.tight_layout()
    plt.show()
