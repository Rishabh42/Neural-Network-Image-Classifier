from torchvision import datasets, transforms


def load_fashion_mnist(data_dirname):
    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.5,), (0.5,), )])

    trainset = datasets.FashionMNIST(data_dirname, download=True, train=True)
    testset = datasets.FashionMNIST(data_dirname, download=True, train=False)

    return trainset, testset


def load_stl10(data_dirname):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,), )])

    trainset = datasets.STL10(data_dirname, download=True, train=True, transform=transform)
    testset = datasets.STL10(data_dirname, download=True, train=False, transform=transform)

    return trainset, testset


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_wine_dataset():

    wine_df = pd.read_csv('./data/wine.csv')
    one_hot_wine_classes = pd.get_dummies(wine_df['class']).to_numpy(dtype=int)

    X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(wine_df.drop(['class'], axis=1).to_numpy(),
                                                                            one_hot_wine_classes, test_size=0.1)

    scalar = StandardScaler().fit(X_wine_train)
    X_wine_train = scalar.transform(X_wine_train)
    X_wine_test = scalar.transform(X_wine_test)

    return X_wine_train, y_wine_train