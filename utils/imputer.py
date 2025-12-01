from abc import ABC, abstractmethod

import torch


class ImputerStrategy(ABC):
    @abstractmethod
    def impute(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MeanImageImputer(ImputerStrategy):
    def impute(self, img: torch.Tensor) -> torch.Tensor:
        x = img.clone()
        if x.dim() == 3:  # C, H, W
            C = x.size(0)
            for c in range(C):
                self._impute_channel(x[c])
        elif x.dim() == 4:  # B, C, H, W
            B, C = x.size(0), x.size(1)
            for b in range(B):
                for c in range(C):
                    self._impute_channel(x[b, c])
        return x

    def _impute_channel(self, channel):
        mean_val = torch.nanmean(channel)
        if torch.isnan(mean_val):
            mean_val = 0.0
        channel[torch.isnan(channel)] = mean_val


class KNNImageImputer(ImputerStrategy):
    def __init__(self, k=3):
        self.k = k

    def impute(self, img: torch.Tensor) -> torch.Tensor:
        x = img.clone()
        if x.dim() == 3:
            C = x.size(0)
            for c in range(C):
                x[c] = self._spatial_knn_2d(x[c])
        elif x.dim() == 4:
            B, C = x.size(0), x.size(1)
            for b in range(B):
                for c in range(C):
                    x[b, c] = self._spatial_knn_2d(x[b, c])
        return x

    def _spatial_knn_2d(self, channel):
        H, W = channel.shape
        device = channel.device
        mask_nan = torch.isnan(channel)
        mask_valid = ~mask_nan

        if not mask_nan.any(): return channel
        if not mask_valid.any(): return torch.zeros_like(channel)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_coords = torch.stack([y_coords, x_coords], dim=-1).float()

        valid_coords = grid_coords[mask_valid]
        missing_coords = grid_coords[mask_nan]
        valid_values = channel[mask_valid]

        dists = torch.cdist(missing_coords, valid_coords)

        k_actual = min(self.k, valid_values.size(0))
        _, nearest_indices = torch.topk(dists, k_actual, dim=1, largest=False)

        neighbor_values = valid_values[nearest_indices]
        imputed_vals = neighbor_values.mean(dim=1)

        out_channel = channel.clone()
        out_channel[mask_nan] = imputed_vals
        return out_channel
