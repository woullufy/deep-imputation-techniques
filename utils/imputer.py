import torch


def mean_impute_image(img):
    x = img.clone()

    if x.dim() == 3:
        # C, H, W
        C = x.size(0)
        for c in range(C):
            channel = x[c]
            mean_val = torch.nanmean(channel)
            channel[torch.isnan(channel)] = mean_val

    elif x.dim() == 4:
        # B, C, H, W
        B, C = x.size(0), x.size(1)
        for b in range(B):
            for c in range(C):
                channel = x[b, c]
                mean_val = torch.nanmean(channel)
                channel[torch.isnan(channel)] = mean_val

    return x


def knn_impute_image(img, k=3):
    x = img.clone()

    if x.dim() == 3:
        # C, H, W
        C = x.size(0)
        for c in range(C):
            x[c] = _spatial_knn_2d(x[c], k)

    elif x.dim() == 4:
        # B, C, H, W
        B, C = x.size(0), x.size(1)
        for b in range(B):
            for c in range(C):
                x[b, c] = _spatial_knn_2d(x[b, c], k)

    return x


def _spatial_knn_2d(channel, k):
    H, W = channel.shape
    device = channel.device

    mask_nan = torch.isnan(channel)
    mask_valid = ~mask_nan

    # Return if no nan pixels present
    if not mask_nan.any():
        return channel

    # Return if all pixels are nan
    if not mask_valid.any():
        return torch.zeros_like(channel)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    grid_coords = torch.stack([y_coords, x_coords], dim=-1).float()

    # Get all the coordinates
    valid_coords = grid_coords[mask_valid]
    missing_coords = grid_coords[mask_nan]

    # Get valid values
    valid_values = channel[mask_valid]

    # Compute the distances between every missing pixel and every valid pixel
    dists = torch.cdist(missing_coords, valid_coords)

    # Get knn pixels
    k_actual = min(k, valid_values.size(0))
    _, nearest_indices = torch.topk(dists, k_actual, dim=1, largest=False)

    neighbor_values = valid_values[nearest_indices]
    imputed_vals = neighbor_values.mean(dim=1)

    # Fill the missing values
    out_channel = channel.clone()
    out_channel[mask_nan] = imputed_vals

    return out_channel
