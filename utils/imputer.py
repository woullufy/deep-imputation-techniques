from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


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


class SklearnGMMImageImputer(ImputerStrategy):
    def __init__(self, n_components=10, ink_threshold=0.1):
        self.n_components = n_components
        self.ink_threshold = ink_threshold
        self.model = None

    def impute(self, img: torch.Tensor) -> torch.Tensor:
        x = img.clone()
        if x.dim() == 3:  # C, H, W
            C = x.size(0)
            for c in range(C):
                x[c] = self._spatial_gmm_2d(x[c])
        elif x.dim() == 4:  # B, C, H, W
            B, C = x.size(0), x.size(1)
            for b in range(B):
                for c in range(C):
                    x[b, c] = self._spatial_gmm_2d(x[b, c])
        return x

    def _spatial_gmm_2d(self, channel: torch.Tensor) -> torch.Tensor:
        device = channel.device

        img_np = channel.detach().cpu().numpy()
        H, W = img_np.shape

        mask_nan = np.isnan(img_np)

        if not mask_nan.any():
            return channel

        y_valid, x_valid = np.where((img_np > self.ink_threshold) & ~mask_nan)
        X_train = np.column_stack([x_valid, y_valid])

        if X_train.shape[0] < self.n_components:
            zeros = torch.zeros_like(channel)
            return zeros

        self.model = GaussianMixture(n_components=self.n_components, covariance_type='full')
        self.model.fit(X_train)

        y_missing, x_missing = np.where(mask_nan)

        if len(y_missing) == 0:
            return channel

        X_missing = np.column_stack([x_missing, y_missing])

        log_density = self.model.score_samples(X_missing)
        density = np.exp(log_density)

        log_density_train = self.model.score_samples(X_train)
        ref_density = np.percentile(np.exp(log_density_train), 50)

        imputed_vals_np = np.clip(density / (ref_density + 1e-10), 0, 1)

        out_channel = channel.clone()
        vals_tensor = torch.from_numpy(imputed_vals_np).to(dtype=channel.dtype, device=device)
        out_channel[mask_nan] = vals_tensor
        return out_channel
