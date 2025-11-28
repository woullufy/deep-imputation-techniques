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
