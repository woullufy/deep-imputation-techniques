import torch


class Missingness:
    def __init__(self, device):
        self.device = device

    def mcar(self, img, missing_rate=0.2):
        if img.dim() == 3:
            C, H, W = img.shape
            mask = torch.rand(C, H, W, device=img.device) > missing_rate

        elif img.dim() == 4:
            B, C, H, W = img.shape
            mask = torch.rand(B, C, H, W, device=img.device) > missing_rate

        masked_img = img.clone()
        masked_img[~mask] = float("nan")

        return masked_img, mask
