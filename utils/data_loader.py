from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils.helpers import random_covariance_matrix


def load_fashion_mnist():
    training_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return training_data, test_data


def load_mnist():
    training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return training_data, test_data


def get_raw_data(dataset_name='mnist', device='cpu'):
    if dataset_name.lower() == 'mnist':
        dataset = datasets.MNIST(root="../data", train=True, download=True)
    elif dataset_name.lower() == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root="../data", train=True, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    tensor_x = dataset.data.float().div(255)
    tensor_x = tensor_x.view(-1, 784).to(device)

    labels = dataset.targets.numpy()
    indices = torch.arange(len(tensor_x)).to(device)

    return tensor_x, labels, indices


def get_tabular_data(dataset_name='penguins', device='cpu', **kwargs):
    if dataset_name.lower() == 'penguins':
        X, y, classes = load_penguins(return_numpy=True, encode_labels=True)
    elif dataset_name.lower() == 'iris':
        X, y, classes = load_iris(return_numpy=True, encode_labels=True)
    elif dataset_name.lower() == 'wine':
        X, y, classes = load_wine(return_numpy=True, encode_labels=True)
    elif dataset_name.lower() == 'gaussian':
        X, y, classes = load_gaussian(**kwargs)
    else:
        raise ValueError(f"Unknown tabular dataset: {dataset_name}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tensor_x = torch.from_numpy(X_scaled).float().to(device)
    labels = y
    indices = torch.arange(len(tensor_x)).to(device)

    return tensor_x, labels, indices, classes


def load_iris(return_numpy=False, encode_labels=False):
    return _load_csv(
        filename="iris.csv",
        drop_cols=["id"],
        return_numpy=return_numpy,
        encode_labels=encode_labels,
        target_name="species"
    )


def load_penguins(return_numpy=False, encode_labels=False):
    return _load_csv(
        filename="penguins.csv",
        drop_cols=['id', 'sex', 'island', 'year'],
        return_numpy=return_numpy,
        encode_labels=encode_labels,
        target_name="species"
    )


def load_gaussian(
        n_samples=1000,
        n_features=2,
        *,
        centers=3,
        cluster_std=1.0,
        center_box=(-10.0, 10.0),
        random_state=None,
):
    rng = np.random.default_rng(random_state)

    sizes = [n_samples // centers] * centers
    sizes[0] += n_samples % centers

    X_list = []
    y_list = []

    for i, size in enumerate(sizes):
        position = rng.uniform(low=center_box[0], high=center_box[1], size=(1, n_features))

        cluster_data = rng.normal(scale=cluster_std, size=(size, n_features))
        cluster_data = cluster_data + position

        sigma = random_covariance_matrix(n_features, rng)
        cluster_data = cluster_data @ sigma

        cluster_label = np.full(size, i)

        X_list.append(cluster_data)
        y_list.append(cluster_label)

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    classes = np.arange(centers)

    return X, y, classes


def load_wine(return_numpy=False, encode_labels=False):
    root = Path(__file__).resolve().parents[1]

    red = pd.read_csv(root / "data" / "winequality-red.csv", sep=";")
    white = pd.read_csv(root / "data" / "winequality-white.csv", sep=";")

    red["type"] = "red"
    white["type"] = "white"

    df = pd.concat([red, white], ignore_index=True)

    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if not return_numpy:
        return df

    X = df.drop(columns=["quality", "type"]).to_numpy()
    y = df["type"].to_numpy()

    if encode_labels:
        classes, y = np.unique(y, return_inverse=True)
        return X, y, classes

    return X, y


def _load_csv(filename, drop_cols, return_numpy, encode_labels, target_name):
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / filename, sep=None, engine="python")

    df.columns = df.columns.str.strip()

    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if not return_numpy:
        return df

    X = df.drop(columns=[target_name]).to_numpy()
    y = df[target_name].to_numpy()

    if encode_labels:
        classes, y = np.unique(y, return_inverse=True)
        return X, y, classes

    return X, y
