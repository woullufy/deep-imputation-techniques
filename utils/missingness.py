import random

import numpy as np
import pandas as pd
import torch


class ImageMissingness:
    def _prepare_input(self, img):
        is_batched = img.dim() == 4
        if not is_batched:
            img = img.unsqueeze(0)

        return img, is_batched

    def _finalize_output(self, masked_img, mask, is_batched):
        if not is_batched:
            masked_img = masked_img.squeeze(0)
            mask = mask.squeeze(0)

        return masked_img, mask

    def mcar(self, img, missing_rate=0.2):
        x, is_batched = self._prepare_input(img)
        mask = torch.rand_like(x) > missing_rate

        masked_img = x.clone()
        masked_img[~mask] = float("nan")

        return self._finalize_output(masked_img, mask, is_batched)

    def mar(self, img, alpha=5.0):
        x, is_batched = self._prepare_input(img)

        norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8)
        prob_missing = torch.sigmoid(alpha * (norm - 0.5))

        mask = torch.rand_like(prob_missing) > prob_missing

        masked_img = x.clone()
        masked_img[~mask] = float("nan")

        return self._finalize_output(masked_img, mask, is_batched)

    def mnar(self, img, threshold=0.3, inverse=False):
        x, is_batched = self._prepare_input(img)

        if inverse:
            mask = x <= threshold
        else:
            mask = x >= threshold

        masked_img = x.clone()
        masked_img[~mask] = float("nan")

        return self._finalize_output(masked_img, mask, is_batched)

    def block_missing(self, img, n_blocks=1, min_size=4, max_size=12, missing_rate=0.75):
        x, is_batched = self._prepare_input(img)
        B, C, H, W = x.shape

        mask = torch.ones_like(x, dtype=torch.bool)

        for b in range(B):
            for _ in range(n_blocks):
                size = random.randint(min_size, max_size)

                x0 = random.randint(0, H - size) if H - size >= 0 else 0
                y0 = random.randint(0, W - size) if W - size >= 0 else 0

                temp_block_mask = torch.rand(C, size, size, device=x.device) > missing_rate

                mask[b, :, x0:x0 + size, y0:y0 + size] = temp_block_mask

        masked_img = x.clone()
        masked_img[~mask] = float("nan")

        return self._finalize_output(masked_img, mask, is_batched)

    def row_missing(self, img, num_rows=5):
        x, is_batched = self._prepare_input(img)
        B, C, H, W = x.shape

        rows = torch.randperm(H)[:num_rows]

        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, :, rows, :] = False

        masked_img = x.clone()
        masked_img[~mask] = float("nan")

        return self._finalize_output(masked_img, mask, is_batched)

    def col_missing(self, img, num_cols=5):
        x, is_batched = self._prepare_input(img)
        B, C, H, W = x.shape

        cols = torch.randperm(W)[:num_cols]

        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, :, :, cols] = False

        masked_img = x.clone()
        masked_img[~mask] = float("nan")

        return self._finalize_output(masked_img, mask, is_batched)

    def salt_pepper(self, img, amount=0.1):
        x, is_batched = self._prepare_input(img)

        out = x.clone()
        noise = torch.rand_like(out)

        pepper_mask = noise < amount / 2
        out[pepper_mask] = torch.min(x)

        salt_mask = noise > 1 - amount / 2
        out[salt_mask] = torch.max(x)

        noise_mask = pepper_mask | salt_mask

        return self._finalize_output(out, noise_mask, is_batched)

    def apply_corruption(self, x, corruption_type, **kwargs):
        is_flat = x.dim() == 2
        if is_flat:
            B, L = x.shape
            side = int(L ** 0.5)
            img = x.view(B, 1, side, side)
        else:
            img = x

        if hasattr(self, corruption_type):
            corruption_func = getattr(self, corruption_type)
            noisy_img, mask = corruption_func(img, **kwargs)
        else:
            noisy_img, mask = img, torch.zeros_like(img, dtype=torch.bool)

        if is_flat:
            noisy_img = noisy_img.view(B, -1)
            mask = mask.view(B, -1)

        return noisy_img, mask


class TabularMissingness:
    def _prepare_input(self, data):
        is_df = isinstance(data, pd.DataFrame)
        df = data.copy() if is_df else pd.DataFrame(data)
        return df, is_df

    def _finalize_output(self, df, is_df):
        return df if is_df else df.to_numpy()

    def mcar(self, data, frac=0.1, columns=None, random_state=None):
        rng = np.random.default_rng(random_state)
        df, is_df = self._prepare_input(data)

        if columns is None:
            mask = rng.random(df.shape) < frac
            df = df.mask(mask)
        else:
            cols = [columns] if isinstance(columns, (str, int)) else columns
            for col in cols:
                mask = rng.random(len(df)) < frac
                df.loc[mask, col] = np.nan
        return self._finalize_output(df, is_df)

    def mar(self, data, frac=0.1, dep_col=None, miss_col=None, random_state=None):
        rng = np.random.default_rng(random_state)
        df, is_df = self._prepare_input(data)

        if dep_col is None or miss_col is None:
            raise ValueError("Dependent column and missing columns cannot be None")

        names = list(df.columns)
        d_idx = [names.index(x) if isinstance(x, str) else x for x in
                 ([dep_col] if isinstance(dep_col, (str, int)) else dep_col)]
        m_idx = [names.index(x) if isinstance(x, str) else x for x in
                 ([miss_col] if isinstance(miss_col, (str, int)) else miss_col)]

        score = df.iloc[:, d_idx].rank(method="average").mean(axis=1)
        high_mask = score > score.median()

        for col in m_idx:
            p = frac.get(col, 0.1) if isinstance(frac, dict) else frac
            mask = high_mask & (rng.random(len(df)) < p)
            df.iloc[mask.values, col] = np.nan
        return self._finalize_output(df, is_df)

    def mnar(self, data, frac=0.1, random_state=None):
        rng = np.random.default_rng(random_state)
        df, is_df = self._prepare_input(data)

        for col in df.columns:
            mask = (df[col] > df[col].median()) & (rng.random(len(df)) < frac)
            df.loc[mask, col] = np.nan
        return self._finalize_output(df, is_df)

    def apply_stratified(self, X, y, method="mcar", **kwargs):
        df, is_df = self._prepare_input(X)
        chunks = []

        for label in np.unique(y):
            subset = df.iloc[y == label].copy()
            out = getattr(self, method)(subset, **kwargs)

            if isinstance(out, np.ndarray):
                out = pd.DataFrame(out, index=subset.index, columns=df.columns)
            chunks.append(out)

        res = pd.concat(chunks).sort_index()
        return self._finalize_output(res, is_df)
