import torch
import random


class Missingness:
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
