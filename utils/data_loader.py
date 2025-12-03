import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


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
