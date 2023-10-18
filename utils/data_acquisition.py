from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder

def load_and_preprocess_data(data_dirname, dataset_name, normalize=True, mlp=True):
    # define a transform to normalize the data
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.ToTensor()

    if dataset_name == 'F_MNIST':
        trainset = datasets.FashionMNIST(data_dirname, download=True, train=True, transform=transform)
        testset = datasets.FashionMNIST(data_dirname, download=True, train=False, transform=transform)
    elif dataset_name == 'CIFAR10':
        trainset = datasets.CIFAR10(data_dirname, download=True, train=True, transform=transform)
        testset = datasets.CIFAR10(data_dirname, download=True, train=False, transform=transform)

    # flatten and scale 
    if mlp:
        X_train = trainset.data.numpy().reshape(-1, 28*28) / 255.0
        y_train = trainset.targets.numpy()
        X_test = testset.data.numpy().reshape(-1, 28*28) / 255.0
        y_test = testset.targets.numpy()
    else:
        return trainset, testset
    
    # one-hot encode
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_oh = encoder.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train_oh, y_test_oh