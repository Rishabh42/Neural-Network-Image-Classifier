import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder


def load_and_preprocess_data(data_dirname, dataset_name, normalize=True, mlp=True):
    """Loads either CIFAR10 or FashionMNIST"""
    assert dataset_name in ['CIFAR10', 'F_MNIST'], 'Enter a valid dataset name'

    if dataset_name == 'F_MNIST':
        trainset = datasets.FashionMNIST(data_dirname, download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.FashionMNIST(data_dirname, download=True, train=False, transform=transforms.ToTensor())

        X_train = torch.Tensor(trainset.data.numpy())
        y_train = torch.Tensor(trainset.targets)
        X_test = torch.Tensor(testset.data.numpy())
        y_test = torch.Tensor(testset.targets)

        X_train = X_train[:, None, :, :]    # Create an image channel for DataLoader
        X_test = X_test[:, None, :, :]

    else:
        trainset = datasets.CIFAR10(data_dirname, download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.CIFAR10(data_dirname, download=True, train=False, transform=transforms.ToTensor())

        X_train = torch.Tensor(trainset.data)
        y_train = torch.Tensor(trainset.targets)
        X_test = torch.Tensor(testset.data)
        y_test = torch.Tensor(testset.targets)

        X_train = torch.transpose(X_train, 1, 3)    # Put image channels first for DataLoader
        X_test = torch.transpose(X_test, 1, 3)

    if normalize:
        X_train = 2 * (X_train / 255.) - 1
        X_test = 2 * (X_test / 255.) - 1

    if mlp:
        X_train = np.array(X_train).reshape(X_train.shape[0], -1)
        X_test = np.array(X_test).reshape(X_test.shape[0], -1)

        # one-hot encode
        encoder = OneHotEncoder(sparse_output=False, categories='auto')
        y_train_oh = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
        y_test_oh = encoder.transform(np.array(y_test).reshape(-1, 1))

        return X_train, X_test, y_train_oh, y_test_oh

    return X_train, X_test, y_train, y_test