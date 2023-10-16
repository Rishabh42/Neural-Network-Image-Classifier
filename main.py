import json
import numpy as np
from utils.data_acquisition import load_fashion_mnist
from models.mlp import MLP
from utils.activations import ReLU, Softmax, Logistic, Tanh
from utils.loss_functions import CrossEntropyLoss
from models.optimizers import StochasticGradientDescent, evaluate_acc
import matplotlib.pyplot as plt

F_MNIST_DATA_DIR = './data/F_MNIST_data'
STL_10_DATA_DIR = './data/STL_10_data'

SEED = 0
np.random.seed(SEED)


def experiment_1(X_train, y_train, X_test, y_test, plot_fname='./out/exp1/test.png', test_acc_fname='./out/exp1/test.txt'):

    from utils.weight_initializers import all_zeros, uniform, gaussian, xavier, kaiming
    initializers = {'Zeros': all_zeros, 'Uniform': uniform, 'Gaussian': gaussian, 'Xavier': xavier, 'Kaiming': kaiming}
    test_accuracies = {}

    layer_sizes = [X_train.shape[1], y_train.shape[1]]
    activations = [Softmax()]
    optimizer_kwargs = {'batch_size': 256, 'n_epochs': 100, 'verbose': True, 'record_loss': True}

    plt.figure(figsize=(8.25, 3))

    for initalizer_name, initializer in initializers.items():

        mlp = MLP(layer_sizes, activations, weight_initializer=initializer)
        optimizer = StochasticGradientDescent(**optimizer_kwargs)
        mlp.fit(X_train, y_train, optimizer=optimizer, loss_fn=CrossEntropyLoss())

        # Plot training loss
        t = [_ for _ in range(optimizer.loss_history.shape[0])]
        plt.plot(t, optimizer.loss_history, alpha=0.7, linewidth=0.8, label=initalizer_name)

        # Get test accuracy
        test_accuracies[initalizer_name] = evaluate_acc(y_test, mlp.predict(X_test))

    print(test_accuracies)
    with open(test_acc_fname, 'w') as test_acc_file:
        test_acc_file.write(json.dumps(test_accuracies))
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('Training Loss')
    plt.tight_layout()
    plt.savefig(plot_fname, dpi=400)


def experiment_2(X_train, y_train, X_test, y_test, plot_fname='./out/exp2/test.png', test_acc_fname='./out/exp2/test.txt'):
    from utils.weight_initializers import kaiming

    layer_sizes_list = [[X_train.shape[1], y_train.shape[1]],
                        [X_train.shape[1], 128, y_train.shape[1]],
                        [X_train.shape[1], 128, 128, y_train.shape[1]]]

    activations_list = [[Softmax()],
                        [ReLU(), Softmax()],
                        [ReLU(), ReLU(), Softmax()]]

    models = [MLP(layer_sizes_list[i], activations_list[i], kaiming) for i in range(len(layer_sizes_list))]

    optimizer_kwargs = {'batch_size': 256, 'n_epochs': 10, 'verbose': True, 'record_loss': True}
    optimizer = StochasticGradientDescent(**optimizer_kwargs)
    test_accuracies = {}

    plt.figure(figsize=(8.25, 3))
    for i, mlp in enumerate(models):

        mlp.fit(X_train, y_train, optimizer=optimizer, loss_fn=CrossEntropyLoss())

        # Plot training loss
        label = r'$n_{h}=' + str(i) + r'$'
        t = [_ for _ in range(optimizer.loss_history.shape[0])]
        plt.plot(t, optimizer.loss_history, alpha=0.7, linewidth=0.8, label=label)

        # Get test accuracy
        test_accuracies[i] = evaluate_acc(y_test, mlp.predict(X_test))

    print(test_accuracies)
    with open(test_acc_fname, 'w') as test_acc_file:
        test_acc_file.write(json.dumps(test_accuracies))

    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('Training Loss')
    plt.tight_layout()
    plt.savefig(plot_fname, dpi=400)


def experiment_3(X_train, y_train, X_test, y_test, plot_fname='./out/exp3/test.png', test_acc_fname='./out/exp3/test.txt'):
    from utils.weight_initializers import xavier, kaiming

    # six experiments: 2 initializers, three activations.
    layer_sizes = [X_train.shape[1], 128, 128, y_train.shape[1]]

    optimizer_kwargs = {'batch_size': 256, 'n_epochs': 10, 'verbose': True, 'record_loss': True}
    optimizer = StochasticGradientDescent(**optimizer_kwargs)

    test_accuracies = {}
    tests = {'Relu, Xa': {'activations': [ReLU(), ReLU(), Softmax()], 'weight_initializer': xavier},
             'Relu, Ka': {'activations': [ReLU(), ReLU(), Softmax()], 'weight_initializer': kaiming},
             'Logistic, Xa': {'activations': [Logistic(), Logistic(), Softmax()], 'weight_initializer': xavier},
             'Logistic, Ka': {'activations': [Logistic(), Logistic(), Softmax()], 'weight_initializer': kaiming},
             'Tanh, Xa': {'activations': [Tanh(), Logistic(), Softmax()], 'weight_initializer': xavier},
             'Tanh, Ka': {'activations': [Tanh(), Logistic(), Softmax()], 'weight_initializer': kaiming}}

    plt.figure(figsize=(8.25, 3))
    for test_name, test_params in tests.items():
        mlp = MLP(layer_sizes, **test_params)
        mlp.fit(X_train, y_train, optimizer=optimizer, loss_fn=CrossEntropyLoss())

        # Plot training loss
        t = [_ for _ in range(optimizer.loss_history.shape[0])]
        plt.plot(t, optimizer.loss_history, alpha=0.7, linewidth=0.8, label=test_name)

        # Get test accuracy
        test_accuracies[test_name] = evaluate_acc(y_test, mlp.predict(X_test))

    print(test_accuracies)
    with open(test_acc_fname, 'w') as test_acc_file:
        test_acc_file.write(json.dumps(test_accuracies))

    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('Training Loss')
    plt.tight_layout()
    plt.savefig(plot_fname, dpi=400)


def experiment_4(X_train, y_train, X_test, y_test, plot_fname='./out/exp4/test.png', test_acc_fname='./out/exp4/test.txt'):
    from utils.weight_initializers import kaiming

    # six experiments: 2 initializers, three activations.
    layer_sizes = [X_train.shape[1], 128, 128, y_train.shape[1]]
    activations = [ReLU(), ReLU(), Softmax()]
    mlp = MLP(layer_sizes, activations, weight_initializer=kaiming)

    optimizer_kwargs = {'batch_size': 256, 'n_epochs': 50, 'verbose': True, 'record_loss': True}

    test_accuracies = {}
    lambds_1 = [0.1, 0.01, 0.001]   # param for l1 regression
    lambds_2 = [0.1, 0.01, 0.001]   # param for l2 regression
    tests = {'None': [{'regularization': None}],
             'L1': [{'regularization': 'l1', 'lambd': lambd} for lambd in lambds_1],
             'L2': [{'regularization': 'l2', 'lambd': lambd} for lambd in lambds_2]}

    plt.figure(figsize=(8.25, 3))
    for regularization_type, test in tests.items():
        if regularization_type == 'None':
            optimizer = StochasticGradientDescent(**optimizer_kwargs)
            mlp.fit(X_train, y_train, optimizer=optimizer, loss_fn=CrossEntropyLoss())

            # Plot training loss
            t = [_ for _ in range(optimizer.loss_history.shape[0])]
            plt.plot(t, optimizer.loss_history, alpha=0.7, linewidth=0.8, label=regularization_type)

            # Get test accuracy
            test_accuracies[regularization_type] = evaluate_acc(y_test, mlp.predict(X_test))

        else:
            for reg_params in test:
                optimizer_kwargs = {'batch_size': 256, 'n_epochs': 50, 'verbose': True, 'record_loss': True}
                optimizer = StochasticGradientDescent(**optimizer_kwargs, **reg_params)
                mlp.fit(X_train, y_train, optimizer=optimizer, loss_fn=CrossEntropyLoss())

                # Plot training loss
                label = regularization_type + r', $\lambda=' + str(reg_params['lambd']) + r'$'
                t = [_ for _ in range(optimizer.loss_history.shape[0])]
                plt.plot(t, optimizer.loss_history, alpha=0.7, linewidth=0.8, label=label)

                # Get test accuracy
                test_accuracies[label] = evaluate_acc(y_test, mlp.predict(X_test))

    print(test_accuracies)
    with open(test_acc_fname, 'w') as test_acc_file:
        test_acc_file.write(json.dumps(test_accuracies))
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('Training Loss')
    plt.tight_layout()
    plt.savefig(plot_fname, dpi=400)


if __name__=='__main__':

    f_mnist_trainset, f_mnist_testset = load_fashion_mnist(F_MNIST_DATA_DIR)
    #stl10_trainset, stl10_testset = load_fashion_mnist(STL_10_DATA_DIR)
    X_train = f_mnist_trainset.data.numpy().reshape(-1, 28*28)
    y_train = f_mnist_trainset.targets.numpy().reshape(-1, 1)

    X_test = f_mnist_testset.data.numpy().reshape(-1, 28*28)
    y_test = f_mnist_testset.targets.numpy().reshape(-1, 1)


    X_train  = X_train / 255 # scale the image data
    X_test  = X_test / 255

    n_train_samples = 3000  # remove for real experiments
    n_test_samples = 500    # remove for real experiments

    X_train = X_train[:n_train_samples]
    y_train = y_train[:n_train_samples]
    X_test = X_test[:n_test_samples]
    y_test = y_test[:n_test_samples]

    # create one hot encoding
    n_classes = y_train.max() + 1
    y_oh = np.eye(n_classes)[y_train].reshape(y_train.shape[0], -1)
    y_train = y_oh

    n_classes = y_test.max() + 1
    y_oh = np.eye(n_classes)[y_test].reshape(y_test.shape[0], -1)
    y_test = y_oh

    experiment_1(X_train, y_train, X_test, y_test)
