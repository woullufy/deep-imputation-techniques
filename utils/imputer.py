from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture


class ImputerStrategy(ABC):
    @abstractmethod
    def impute(self, x):
        pass


class MeanImageImputer(ImputerStrategy):
    def impute(self, img):
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

    def impute(self, img):
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

    def impute(self, img):
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

    def _spatial_gmm_2d(self, channel):
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


class BatchedSpatialGMM:
    def init(self, n_components=10, n_iters=25, tol=1e-3, reg_covar=1e-4):
        self.K = n_components
        self.n_iters = n_iters
        self.tol = tol
        self.reg_covar = reg_covar

    def fit(self, X, mask=None):
        B, N, D = X.shape
        device = X.device

        if mask is None:
            mask = torch.ones(B, N, device=device)

        rand_vals = torch.rand(B, N, device=device) * mask
        _, top_k_idx = torch.topk(rand_vals, self.K, dim=1)

        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        self.mu = X[batch_idx, top_k_idx]  # (B, K, 2)

        valid_counts = mask.sum(dim=1).view(B, 1, 1) + 1e-10
        global_mean = (X * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_counts
        global_var = ((X - global_mean) ** 2 * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_counts

        avg_var = global_var.mean(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1, 1)

        # Initialize Sigma
        eye = torch.eye(D, device=device).view(1, 1, D, D)
        self.sigma = eye * avg_var
        self.sigma = self.sigma.expand(B, self.K, D, D).clone()

        self.pi = torch.ones(B, self.K, device=device) / self.K

        # --- EM Loop ---
        for i in range(self.n_iters):
            log_resp = self._estimate_log_prob(X)
            log_resp = log_resp + torch.log(mask.unsqueeze(-1) + 1e-10)

            log_prob_norm = torch.logsumexp(log_resp, dim=2, keepdim=True)
            resp = torch.exp(log_resp - log_prob_norm)  # (B, N, K)
            resp = resp * mask.unsqueeze(-1)

            # M-Step
            N_k = resp.sum(dim=1) + 1e-10
            self.pi = N_k / N_k.sum(dim=1, keepdim=True)
            self.mu = torch.bmm(resp.transpose(1, 2), X) / N_k.unsqueeze(-1)

            diff = X.unsqueeze(2) - self.mu.unsqueeze(1)
            sigma_new = torch.einsum('bnk,bnki,bnkjbkij', resp, diff, diff)
            sigma_new = sigma_new / N_k.unsqueeze(-1).unsqueeze(-1)

            eye = torch.eye(D, device=device).expand(B, self.K, D, D)
            self.sigma = sigma_new + self.reg_covar * eye

    def predict_density(self, X):
        log_prob = self._estimate_log_prob(X)
        return torch.exp(torch.logsumexp(log_prob, dim=2))

    def _estimate_log_prob(self, X):
        B, N, _ = X.shape
        D = 2
        X_exp = X.unsqueeze(2)
        mu_exp = self.mu.unsqueeze(1)
        diff = X_exp - mu_exp

        precision = torch.linalg.inv(self.sigma)
        log_det = torch.logdet(self.sigma)

        diff_prec = torch.einsum('bnki,bkijbnkj', diff, precision)
        mahalanobis = (diff_prec * diff).sum(dim=-1)

        log_2pi = D * np.log(2 * np.pi)
        log_prob = -0.5 * (log_2pi + log_det.unsqueeze(1) + mahalanobis)

        return log_prob + torch.log(self.pi.unsqueeze(1) + 1e-10)


class TESTImputer(ImputerStrategy):
    def init(self, n_components=10, ink_threshold=0.1, n_iters=25):
        self.n_components = n_components
        self.ink_threshold = ink_threshold
        self.n_iters = n_iters

    def impute(self, img):
        x = img.clone()
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0)
        elif original_dim == 4:
            B, C, H, W = x.shape
            x = x.view(B * C, H, W)

        imputed_batch = self._impute_batch(x)

        if original_dim == 2:
            return imputed_batch.squeeze(0)
        elif original_dim == 4:
            return imputed_batch.view(B, C, H, W)
        return imputed_batch

    def _impute_batch(self, imgs):
        B, H, W = imgs.shape
        device = imgs.device

        mask_nan = torch.isnan(imgs)
        mask_valid = ~mask_nan & (imgs > self.ink_threshold)

        batch_idx, y_coord, x_coord = torch.where(mask_valid)
        counts = torch.bincount(batch_idx, minlength=B)
        max_points = counts.max().item()

        if max_points < self.n_components:
            return torch.nan_to_num(imgs, 0.0)

        X_train = torch.zeros(B, max_points, 2, device=device)
        train_mask = torch.zeros(B, max_points, device=device)

        for b in range(B):
            valid_b = torch.stack([x_coord[batch_idx == b], y_coord[batch_idx == b]], dim=1)
            n_p = valid_b.shape[0]
            if n_p > 0:
                X_train[b, :n_p] = valid_b.float()
                train_mask[b, :n_p] = 1.0

        model = BatchedSpatialGMM(n_components=self.n_components, n_iters=self.n_iters)
        model.fit(X_train, mask=train_mask)

        batch_idx_m, y_m, x_m = torch.where(mask_nan)
        counts_m = torch.bincount(batch_idx_m, minlength=B)
        max_missing = counts_m.max().item()

        if max_missing == 0: return imgs

        X_missing = torch.zeros(B, max_missing, 2, device=device)

        for b in range(B):
            miss_b = torch.stack([x_m[batch_idx_m == b], y_m[batch_idx_m == b]], dim=1)
            n_m = miss_b.shape[0]
            if n_m > 0:
                X_missing[b, :n_m] = miss_b.float()

        densities = model.predict_density(X_missing)
        train_densities = model.predict_density(X_train)

        masked_densities = train_densities.clone()
        masked_densities[train_mask == 0] = float('nan')
        ref_density = torch.nanquantile(masked_densities, 0.5, dim=1)
        ref_density[torch.isnan(ref_density)] = 1.0

        imputed_vals = densities / (ref_density.unsqueeze(1) + 1e-10)
        imputed_vals = torch.clamp(imputed_vals, 0, 1)

        out_imgs = imgs.clone()
        for b in range(B):
            n_m = counts_m[b]
            if n_m > 0:
                y_locs = y_m[batch_idx_m == b]
                x_locs = x_m[batch_idx_m == b]
                vals = imputed_vals[b, :n_m]
                out_imgs[b, y_locs, x_locs] = vals.type(imgs.dtype)

        return out_imgs


class MeanTabularImputer(ImputerStrategy):
    def impute(self, x):
        if isinstance(x, pd.DataFrame):
            return x.fillna(x.mean())

        out = np.copy(x)
        col_means = np.nanmean(out, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)

        inds = np.where(np.isnan(out))
        out[inds] = np.take(col_means, inds[1])
        return out


class MedianTabularImputer(ImputerStrategy):
    def impute(self, x):
        if isinstance(x, pd.DataFrame):
            return x.fillna(x.median())

        out = np.copy(x)
        col_medians = np.nanmedian(out, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)

        inds = np.where(np.isnan(out))
        out[inds] = np.take(col_medians, inds[1])
        return out


class KNNTabularImputer(ImputerStrategy):
    def __init__(self, k=3):
        self.k = k

    def impute(self, x):
        is_df = isinstance(x, pd.DataFrame)
        data = x.to_numpy() if is_df else np.copy(x)

        mask = np.isnan(data)
        has_nan = mask.any(axis=1)
        if not has_nan.any():
            return x

        col_means = np.nanmean(data, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        ref_data = np.where(mask, col_means, data)

        valid_rows = ref_data[~has_nan]
        if len(valid_rows) == 0:
            return pd.DataFrame(ref_data, columns=x.columns) if is_df else ref_data

        nan_indices = np.where(has_nan)[0]
        for i in nan_indices:
            row = ref_data[i:i + 1]
            distances = cdist(row, valid_rows, metric='euclidean')[0]

            k_neighbors = np.argpartition(distances, min(self.k, len(distances) - 1))[:self.k]

            missing_cols = mask[i]
            neighbor_values = valid_rows[k_neighbors][:, missing_cols]
            data[i, missing_cols] = np.mean(neighbor_values, axis=0)

        return pd.DataFrame(data, index=x.index, columns=x.columns) if is_df else data
