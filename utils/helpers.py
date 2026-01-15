import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def random_covariance_matrix(n, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = rng.uniform(0.4, 4.0, size=n)
    cov = Q @ np.diag(eigenvalues) @ Q.T

    return cov


def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)

    for i in range(len(y_true)):
        cost_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    accuracy = cost_matrix[row_ind, col_ind].sum() / len(y_true)
    return accuracy
