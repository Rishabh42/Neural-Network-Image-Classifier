import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from itertools import product
import pandas as pd

from utils.weight_initializers import all_zeros, uniform, gaussian, xavier_uniform, kaiming
from utils.activations import ReLU, Logistic, Softplus, Softmax
from utils.loss_functions import CrossEntropyLoss
from utils.optimizers import SGD, Adam
from models.mlp import MLP
from models.cnn import CNN
from utils.data_acquisition import load_and_preprocess_data
from utils.plotting import compare_training_histories, plot_training_history, compare_accuracies


def exp1(optimizer_kwargs, optimizer_name, filepath='./out/exp1/', 
         epochs=100, batch_size=256, verbose=True):
    
    X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST')
    
    initializers = {'Zeros': all_zeros, 'Uniform': uniform, 'Gaussian': gaussian, 'Xavier': xavier_uniform, 'Kaiming': kaiming}
    histories = []
    final_accuracies = [] 

    for name, init in initializers.items():
        print(f"Training with {name} initialization...")
        model = MLP(layer_sizes = [X_train.shape[1], 128, y_train_oh.shape[1]], 
                    act_fn=ReLU(), 
                    loss_fn=CrossEntropyLoss(), 
                    softmax_fn=Softmax(), 
                    weight_initializer=init)
        
        if optimizer_name == 'SGD':
            optimizer = SGD(**optimizer_kwargs)
        elif optimizer_name == 'Adam':
            optimizer = Adam(**optimizer_kwargs)

        history = model.fit(X_train, y_train_oh, optimizer, X_test=X_test, y_test=y_test_oh, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        histories.append(history)
        
        # calc final train/test accuracies
        final_train_acc = np.mean(np.argmax(model.forward(X_train)[-1], axis=1) == np.argmax(y_train_oh, axis=1))
        final_test_acc = np.mean(np.argmax(model.forward(X_test)[-1], axis=1) == np.argmax(y_test_oh, axis=1))
        final_accuracies.append((name, final_train_acc, final_test_acc))

        print(f"Completed training with {name} initialization. Final Train Accuracy: {final_train_acc:.4f}, Final Test Accuracy: {final_test_acc:.4f}\n")

    # save final accuracies
    with open(f'{filepath}/final_accuracies.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)

    # save histories
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)

    # save plots
    compare_training_histories(histories, titles=list(initializers.keys()), 
                               filename=f'{filepath}/training_histories.png', show=False)
    
    compare_accuracies(histories, labels=list(initializers.keys()), plot_train=False,
                       filename=f'{filepath}/accuracies.png', show=False)

    return histories, final_accuracies

def exp2(optimizer_kwargs, optimizer_name,filepath='./out/exp2/', epochs=40, batch_size=256, verbose=True):
    X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST')

    model_architectures = [
        {
            'name': 'No Hidden Layers',
            'layer_sizes': [X_train.shape[1], y_train_oh.shape[1]],
        },
        {
            'name': 'Single Hidden Layer',
            'layer_sizes': [X_train.shape[1], 128, y_train_oh.shape[1]],
        },
        {
            'name': 'Two Hidden Layers',
            'layer_sizes': [X_train.shape[1], 128, 128, y_train_oh.shape[1]],
        },
    ]

    histories = []
    final_accuracies = []

    for architecture in model_architectures:
        print(f"Training {architecture['name']} model...")
        model = MLP(
            layer_sizes=architecture['layer_sizes'],
            act_fn=ReLU(),
            loss_fn=CrossEntropyLoss(),
            softmax_fn=Softmax(),
            weight_initializer=kaiming,
        )

        if optimizer_name == 'SGD':
            optimizer = SGD(**optimizer_kwargs)
        elif optimizer_name == 'Adam':
            optimizer = Adam(**optimizer_kwargs)

        history = model.fit(
            X_train, y_train_oh, optimizer, X_test=X_test, y_test=y_test_oh,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )

        histories.append(history)

        # Calculate and record the final test accuracy
        final_test_acc = np.mean(np.argmax(model.forward(X_test)[-1], axis=1) == np.argmax(y_test_oh, axis=1))
        final_accuracies.append((architecture['name'], final_test_acc))

        print(f"{architecture['name']} Model - Final Test Accuracy: {final_test_acc:.4f}\n")

        #TODO: create plots (later)

    # save histories
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)

    # Save final accuracies
    with open(f'{filepath}/final_accuracies.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)

    # save plots
    model_names = ['No Hidden Layers', 'Single Hidden Layer', 'Two Hidden Layers']

    compare_training_histories(histories, titles=list(model_names), 
                                filename=f'{filepath}/training_histories.png', show=False)
    compare_accuracies(histories, labels=list(model_names), plot_train=False,
                       filename=f'{filepath}/accuracies.png', show=False)

    return histories, final_accuracies

def exp3(optimizer_kwargs, optimizer_name,filepath='./out/exp3/', epochs=50, batch_size=256, verbose=True):
    X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST')

    # Comparing Logistic and Softplus activations with ReLU
    activations = [Logistic(), Softplus(), ReLU()]

    histories = []
    final_accuracies = []

    for activation in activations:
        activation_name = activation.__class__.__name__
        print(f"Training model with {activation_name} activation...")

        model = MLP(
            layer_sizes=[X_train.shape[1], 128, 128, y_train_oh.shape[1]],
            act_fn=activation,
            loss_fn=CrossEntropyLoss(),
            softmax_fn=Softmax(),
            weight_initializer=kaiming,
        )

        if optimizer_name == 'SGD':
            optimizer = SGD(**optimizer_kwargs)
        elif optimizer_name == 'Adam':
            optimizer = Adam(**optimizer_kwargs)

        history = model.fit(
            X_train, y_train_oh, optimizer, X_test=X_test, y_test=y_test_oh,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )

        histories.append(history)

        # Calculate and record the final test accuracy
        final_test_acc = np.mean(np.argmax(model.forward(X_test)[-1], axis=1) == np.argmax(y_test_oh, axis=1))
        final_accuracies.append((activation_name, final_test_acc))

        print(f"{activation_name} Model - Final Test Accuracy: {final_test_acc:.4f}\n")

    # save histories
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)

    # Save final accuracies
    with open(f'{filepath}/final_accuracies_exp3.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)

    # save plots
    activation_models = ["Logistic", "Softplus", "ReLU"]

    compare_training_histories(histories, titles=list(activation_models), 
                                filename=f'{filepath}/training_histories.png', show=False)
    compare_accuracies(histories, labels=list(activation_models), plot_train=False,
                       filename=f'{filepath}/accuracies.png', show=False)

    return histories, final_accuracies

def exp4(filepath='./out/exp4/', epochs=100, batch_size=256, verbose=True):
    X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST')

    regularization_methods = ["None", "l1", "l2"]
    optimizer_kwargs_list = [
        {
        'lr': 0.01, 
        'decay': 1e-6, 
        'momentum': 0.9,
        'regularization': 'None',
        'lambd': 0.001
        },
        {
        'lr': 0.01, 
        'decay': 1e-6, 
        'momentum': 0.9,
        'regularization': 'l1',
        'lambd': 0.001
        },
        {
        'lr': 0.01, 
        'decay': 1e-6, 
        'momentum': 0.9,
        'regularization': 'l2',
        'lambd': 0.001
        },
    ]

    histories = []
    final_accuracies = []

    for optimizer_i in optimizer_kwargs_list:
        print(f"Training {optimizer_i['regularization']} regularization model...")
        model = MLP(
            layer_sizes=[X_train.shape[1], 128, 128, y_train_oh.shape[1]],
            act_fn=ReLU(),
            loss_fn=CrossEntropyLoss(),
            softmax_fn=Softmax(),
            weight_initializer=kaiming,
        )

        optimizer = SGD(**optimizer_i)

        history = model.fit(
            X_train, y_train_oh, optimizer, X_test=X_test, y_test=y_test_oh,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )

        histories.append(history)

        # Calculate and record the final test accuracy
        final_test_acc = np.mean(np.argmax(model.forward(X_test)[-1], axis=1) == np.argmax(y_test_oh, axis=1))
        final_accuracies.append((optimizer_i['regularization'], final_test_acc))

        print(f"{optimizer_i['regularization']} Model - Final Test Accuracy: {final_test_acc:.4f}\n")

        #TODO: create plots (later)

    # save histories
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)

    # Save final accuracies
    with open(f'{filepath}/final_accuracies.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)

    # save plots
    compare_training_histories(histories, titles=list(regularization_methods), 
                                filename=f'{filepath}/training_histories.png', show=False)
    compare_accuracies(histories, labels=list(regularization_methods), plot_train=False,
                       filename=f'{filepath}/accuracies.png', show=False)

    return histories, final_accuracies


def exp6(optimizer_kwargs, filepath='./out/exp6', stride=1, kernel=3, 
         padding=1, epochs=100, batch_size=16, verbose=True):
    """Implement CNN with Pytorch"""

    trainset, testset = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST', normalize=True, mlp=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    histories = []
    final_accuracies = []


def exp6_grid_search(param_grid, filepath='./out/exp6/grid_search', verbose=False):

    X_train, X_test, y_train, y_test = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST', normalize=True, mlp=False)
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    testset = torch.utils.data.TensorDataset(X_test, y_test)

    histories = []
    final_accuracies = []
    all_results = []

    param_combinations = product(*param_grid.values())
    param_names = list(param_grid.keys())

    # grid search
    for params in param_combinations:
        conv1_out, conv2_out, stride, kernel_size, padding, optimizer_name, lr, batch_size, epochs = params
        if optimizer_name == 'SGD':
            optimizer_kwargs = {'lr': lr, 'momentum': 0.9}
        elif optimizer_name == 'Adam':
            optimizer_kwargs = {'lr': lr}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # init model
        model = CNN(input_channels=X_train.shape[1], image_size=X_train.shape[2], conv1_out=conv1_out, conv2_out=conv2_out, stride=stride,
                    kernel_size=kernel_size, padding=padding, optimizer=optimizer_name, **optimizer_kwargs)

        # train
        history = model.fit(trainloader, testloader, epochs=epochs, verbose=True)
        histories.append(history)

        # eval
        _, final_accuracy = model.evaluate(testloader)
        final_accuracies.append(final_accuracy)

        current_run_data = dict(zip(param_names, params))  
        current_run_data['final_accuracy'] = final_accuracy
        all_results.append(current_run_data)

        # Optionally print out the results for each combination of parameters
        if verbose:
            params_dict = dict(zip(param_names, params))
            params_str = ', '.join(f'{key}={value}' for key, value in params_dict.items())
            print(f"Completed training with params: {params_str}. Final Test Accuracy: {final_accuracy:.4f}\n")

    # save results
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)
    with open(f'{filepath}/final_accuracies.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{filepath}/grid_search_results.csv', index=False)
            
    return histories, final_accuracies


if __name__ == '__main__':

    optimizer_kwargs = {
        'lr': 0.01, 
        'decay': 1e-6, 
        'momentum': 0.9,
        'regularization': 'l2',
        'lambd': 0.001
        }
    optimizer = 'SGD'
    batch_size = 256
    epochs = 100

    #exp1(optimizer_kwargs, optimizer_name=optimizer, epochs=epochs, batch_size=batch_size)


    param_grid = {
        'conv1_out': [16, 32],
        'conv2_out': [32, 64],
        'stride': [1],
        'kernel_size': [3, 5],
        'padding': [1, 2],
        'optimizer': ['SGD'],
        'lr': [0.001, 0.01],
        'batch_size': [16, 32, 64],
        'epochs': [5]
    }

    param_grid = {
        'conv1_out': [16],
        'conv2_out': [32],
        'stride': [1],
        'kernel_size': [3],
        'padding': [1],
        'optimizer': ['Adam'],
        'lr': [0.001],
        'batch_size': [16],
        'epochs': [1]
    }

    # exp6_grid_search(param_grid, verbose=True)
    # exp3(optimizer_kwargs,'SGD', verbose=True)
    exp4(verbose=True)

