import numpy as np
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def random_covariance_matrix(n, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = rng.uniform(0.4, 4.0, size=n)
    cov = Q @ np.diag(eigenvalues) @ Q.T

    return cov
