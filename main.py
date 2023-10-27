import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from itertools import product
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.weight_initializers import all_zeros, uniform, gaussian, xavier_uniform, kaiming
from utils.activations import ReLU, Logistic, Tanh, Softmax
from utils.loss_functions import CrossEntropyLoss
from utils.optimizers import SGD, Adam
from models.mlp import MLP
from models.cnn import CNN
from utils.data_acquisition import load_and_preprocess_data, pca_reduce
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
            optimizer_kwargs.pop('momentum', None)
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

def exp2(optimizer_kwargs, optimizer_name, filepath='./out/exp2/', epochs=50, batch_size=256, verbose=True):
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
            optimizer_kwargs.pop('momentum', None)
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

def exp3(optimizer_kwargs, optimizer_name, filepath='./out/exp3/', epochs=50, batch_size=256, verbose=True):
    X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST')

    # Comparing Logistic and Softplus activations with ReLU
    activations = [Logistic(), Tanh(), ReLU()]

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
            optimizer_kwargs.pop('momentum', None)
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
    with open(f'{filepath}/final_accuracies.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)

    # save plots
    activation_models = ["Logistic", "Tanh", "ReLU"]

    compare_training_histories(histories, titles=list(activation_models), 
                                filename=f'{filepath}/training_histories.png', show=False)
    compare_accuracies(histories, labels=list(activation_models), plot_train=False,
                       filename=f'{filepath}/accuracies.png', show=False)

    return histories, final_accuracies

def exp4(optimizer_kwargs, optimizer_name, filepath='./out/exp4/', epochs=50, batch_size=256, verbose=True):
    X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST')

    regularization_methods = ["None", "l1", "l2"]

    histories = []
    final_accuracies = []

    for regularizer in regularization_methods:
        print(f"Training {regularizer} regularization model...")
        optimizer_kwargs['regularization'] = regularizer if regularizer != "None" else None

        if optimizer_name == 'SGD':
            optimizer = SGD(**optimizer_kwargs)
        elif optimizer_name == 'Adam':
            optimizer_kwargs.pop('momentum', None)
            optimizer = Adam(**optimizer_kwargs)

        model = MLP(
            layer_sizes=[X_train.shape[1], 128, 128, y_train_oh.shape[1]],
            act_fn=ReLU(),
            loss_fn=CrossEntropyLoss(),
            softmax_fn=Softmax(),
            weight_initializer=kaiming,
        )

        history = model.fit(
            X_train, y_train_oh, optimizer, X_test=X_test, y_test=y_test_oh,
            epochs=epochs, batch_size=batch_size, verbose=verbose, print_every=5
        )

        histories.append(history)

        # Calculate and record the final test accuracy
        final_test_acc = np.mean(np.argmax(model.forward(X_test)[-1], axis=1) == np.argmax(y_test_oh, axis=1))
        final_accuracies.append((optimizer_kwargs['regularization'], final_test_acc))

        print(f"{optimizer_kwargs['regularization']} Model - Final Test Accuracy: {final_test_acc:.4f}\n")

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

def exp5(optimizer_kwargs, optimizer_name, filepath='./out/exp5/', epochs=50, batch_size=256, verbose=True):
    normalize_data_flags = [True, False]

    histories = []
    final_accuracies = []

    for norm_param in normalize_data_flags:
        X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST', normalize=norm_param)

        print(f"Training ReLU model with norm flag: {norm_param}")
        model = MLP(
            layer_sizes=[X_train.shape[1], 128, 128, y_train_oh.shape[1]],
            act_fn=ReLU(),
            loss_fn=CrossEntropyLoss(),
            softmax_fn=Softmax(),
            weight_initializer=kaiming,
        )

        if optimizer_name == 'SGD':
            optimizer = SGD(**optimizer_kwargs)
        elif optimizer_name == 'Adam':
            optimizer_kwargs.pop('momentum', None)
            optimizer = Adam(**optimizer_kwargs)

        history = model.fit(
            X_train, y_train_oh, optimizer, X_test=X_test, y_test=y_test_oh,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )

        histories.append(history)

        # Calculate and record the final test accuracy
        final_test_acc = np.mean(np.argmax(model.forward(X_test)[-1], axis=1) == np.argmax(y_test_oh, axis=1))
        final_accuracies.append((f"Norm flag {norm_param}", final_test_acc))

        print(f"Model with norm flag {norm_param} - Final Test Accuracy: {final_test_acc:.4f}\n")

    # save histories
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)

    # Save final accuracies
    with open(f'{filepath}/final_accuracies.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)

    # save plots
    data_modes = ["Normalized", "Unnormalized"]

    compare_training_histories(histories, titles=data_modes, 
                                filename=f'{filepath}/training_histories.png', show=False)
    compare_accuracies(histories, labels=data_modes, plot_train=False,
                       filename=f'{filepath}/accuracies.png', show=False)

    return histories, final_accuracies

def exp5_pca(optimizer_kwargs, optimizer_name, ranks, dataset_name, filepath='./out/exp5_pca/', epochs=50, batch_size=256, verbose=True):

    assert ranks[0] == 'Full', 'Full rank baseline must be first'

    histories = []
    final_accuracies = []
    if dataset_name == 'F_MNIST':
        X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST')
    else:
        X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/cifar10_data', dataset_name='CIFAR10', normalize=True, mlp=True)

    for rank in ranks:
        if rank == 'Full':
            X_train_rank, X_test_rank = X_train, X_test
        else:
            X_train_rank, X_test_rank = pca_reduce(X_train, X_test, n_components=rank)

        print(f"Training on data of rank: {rank}, {X_train_rank.shape}")
        model = MLP(
            layer_sizes=[X_train_rank.shape[1], 128, 128, y_train_oh.shape[1]],
            act_fn=ReLU(),
            loss_fn=CrossEntropyLoss(),
            softmax_fn=Softmax(),
            weight_initializer=kaiming,
        )

        if optimizer_name == 'SGD':
            optimizer = SGD(**optimizer_kwargs)
        elif optimizer_name == 'Adam':
            optimizer_kwargs.pop('momentum', None)
            optimizer = Adam(**optimizer_kwargs)

        history = model.fit(
            X_train_rank, y_train_oh, optimizer, X_test=X_test_rank, y_test=y_test_oh,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )

        histories.append(history)

        # Calculate and record the final test accuracy
        final_test_acc = np.mean(np.argmax(model.forward(X_test_rank)[-1], axis=1) == np.argmax(y_test_oh, axis=1))
        final_accuracies.append((f"Rank {rank}", final_test_acc))

        print(f"Model train with rank {rank} data - Final Test Accuracy: {final_test_acc:.4f}\n")

    # save histories
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)

    # Save final accuracies
    with open(f'{filepath}/final_accuracies.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)

    # save plots
    data_modes = [f'Rank {rank}' for rank in ranks[1:]]
    data_modes = ['Full Rank'] + data_modes


    compare_training_histories(histories, titles=data_modes,
                               filename=f'{filepath}/training_histories.png', show=False)
    compare_accuracies(histories, labels=data_modes, plot_train=False,
                       filename=f'{filepath}/accuracies.png', show=False)

    return histories, final_accuracies

def exp6(optimizer_kwargs, filepath='./out/exp6', conv1_out=32, conv2_out=64, stride=1, 
         kernel=3, padding=2, epochs=5, batch_size=16, verbose=True):
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST', normalize=True, mlp=False)
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    testset = torch.utils.data.TensorDataset(X_test, y_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = CNN(input_channels=1, image_size=28, conv1_out=conv1_out, conv2_out=conv2_out, stride=stride, 
                kernel_size=kernel, padding=padding, optimizer='SGD', **optimizer_kwargs)
    
    history = model.fit(trainloader, testloader, epochs=epochs, verbose=verbose, save_every_n_batches=180)
    _, final_accuracy = model.evaluate(testloader)

    # save results
    with open(f'{filepath}/history.pickle', 'wb') as f:
        pickle.dump(history, f)
    with open(f'{filepath}/final_accuracy.pickle', 'wb') as f:
        pickle.dump(final_accuracy, f)

    # save plots
    num_saves_per_epoch = np.floor(len(trainloader) / 180)
    plot_training_history(history, num_saves_per_epoch=num_saves_per_epoch, 
                          filename=f'{filepath}/training_history.png', show=False)

    return history, final_accuracy

def exp7(params_mlp, params_cnn, filepath='./out/exp7', verbose=True):
    histories = []
    final_accuracies = []

    ## MLP
    X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/cifar10_data', dataset_name='CIFAR10', normalize=True, mlp=True)

    # unpack params
    hidden_layer_size = params_mlp['hidden_layer_size']
    epochs = params_mlp['epochs']
    batch_size = params_mlp['batch_size']
    optimizer_kwargs = {'lr': params_mlp['lr'], 
                        'momentum': params_mlp['momentum']}
    optimizer_name = params_mlp['optimizer']
    if optimizer_name == 'SGD':
        optimizer = SGD(**optimizer_kwargs)
    elif optimizer_name == 'Adam':
        optimizer = Adam(**optimizer_kwargs)

    # init model
    model = MLP(layer_sizes=[X_train.shape[1], hidden_layer_size, y_train_oh.shape[1]], 
                act_fn=ReLU(), 
                loss_fn=CrossEntropyLoss(), 
                softmax_fn=Softmax(), 
                weight_initializer=xavier_uniform)
    
    # train
    history = model.fit(X_train, y_train_oh, X_test=X_test, y_test=y_test_oh, optimizer=optimizer, epochs=epochs, batch_size=batch_size, verbose=verbose)
    histories.append(history)

    # eval
    final_accuracy = np.mean(np.argmax(model.forward(X_test)[-1], axis=1) == np.argmax(y_test_oh, axis=1))
    final_accuracies.append(final_accuracy)

    print(f"Completed training MLP. Final Test Accuracy: {final_accuracy:.4f}\n")

    ## CNN
    X_train, X_test, y_train, y_test = load_and_preprocess_data('./data/cifar10_data', dataset_name='CIFAR10', normalize=True, mlp=False)
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    testset = torch.utils.data.TensorDataset(X_test, y_test)

    # unpack params
    conv1_out = params_cnn['conv1_out']
    conv2_out = params_cnn['conv2_out']
    stride = params_cnn['stride']
    kernel_size = params_cnn['kernel_size']
    padding = params_cnn['padding']
    epochs = params_cnn['epochs']
    batch_size = params_cnn['batch_size']
    optimizer_kwargs = {'lr': params_cnn['lr'],
                        'momentum': params_cnn['momentum']}
    optimizer_name = params_cnn['optimizer']

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # init model
    model = CNN(input_channels=3, image_size=32, conv1_out=conv1_out, conv2_out=conv2_out, stride=stride,
                kernel_size=kernel_size, padding=padding, optimizer=optimizer_name, **optimizer_kwargs)
    
    # train
    history = model.fit(trainloader, testloader, epochs=epochs, verbose=verbose, save_every_n_batches=150)
    histories.append(history)

    # eval
    _, final_accuracy = model.evaluate(testloader)
    final_accuracies.append(final_accuracy)

    print(f"Completed training CNN. Final Test Accuracy: {final_accuracy:.4f}\n")

    # save results
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)
    with open(f'{filepath}/final_accuracies.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)
        
    # save plots
    plot_training_history(histories[0], num_saves_per_epoch=1, filename=f'{filepath}/mlp_training_history.png', show=False)
    num_saves_per_epoch = np.floor(len(trainloader) / 150)
    plot_training_history(histories[1], num_saves_per_epoch=num_saves_per_epoch, filename=f'{filepath}/cnn_training_history.png', show=False)

    return histories, final_accuracies

def exp8(lr_sgd, lr_adam, filepath='./out/exp8', conv1_out=32, conv2_out=64, stride=1, 
         kernel=3, padding=2, epochs=5, batch_size=16, verbose=True):

    X_train, X_test, y_train, y_test = load_and_preprocess_data('./data/cifar10_data', dataset_name='CIFAR10', normalize=True, mlp=False)
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    testset = torch.utils.data.TensorDataset(X_test, y_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    histories = []
    final_accuracies = []
    momentums = [0, 0.5, 0.9, 0.99]

    for momentum in momentums:
        print(f"Training with momentum={momentum}...")
        optimizer_kwargs = {'lr': lr_sgd, 'momentum': momentum}
        model = CNN(input_channels=3, image_size=32, conv1_out=conv1_out, conv2_out=conv2_out, stride=stride,
                    kernel_size=kernel, padding=padding, optimizer='SGD', **optimizer_kwargs)
        history = model.fit(trainloader, testloader, epochs=epochs, verbose=verbose, save_every_n_batches=150)
        histories.append(history)

        # eval
        _, final_accuracy = model.evaluate(testloader)
        final_accuracies.append(final_accuracy)

        print(f"Completed training with momentum={momentum}. Final Test Accuracy: {final_accuracy:.4f}\n")

    print(f"Training with Adam...")
    optimizer_kwargs = {'lr': lr_adam}
    model = CNN(input_channels=3, image_size=32, conv1_out=conv1_out, conv2_out=conv2_out, stride=stride,
                kernel_size=kernel, padding=padding, optimizer='Adam', **optimizer_kwargs)
    
    history = model.fit(trainloader, testloader, epochs=epochs, verbose=verbose, save_every_n_batches=150)
    histories.append(history)

    # eval
    _, final_accuracy = model.evaluate(testloader)
    final_accuracies.append(final_accuracy)

    print(f"Completed training with Adam. Final Test Accuracy: {final_accuracy:.4f}\n")

    # save results
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)
    with open(f'{filepath}/final_accuracies.pickle', 'wb') as f:
        pickle.dump(final_accuracies, f)

    # save plots
    titles = [f'SGD with $\\beta={momentum}$' for momentum in momentums] + ['Adam']
    num_saves_per_epoch = np.floor(len(trainloader) / 150)
    compare_training_histories(histories, titles=titles, filename=f'{filepath}/training_histories.png', 
                               num_saves_per_epoch=num_saves_per_epoch, show=False)

    return histories, final_accuracies


def mlp_grid_search(param_grid, filepath='./out/grid_search/mlp', verbose=False, RUN_EXP=False):

    if RUN_EXP:
        X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST',
                                                                    normalize=True, mlp=True)
        X_train, X_val, y_train_oh, y_val_oh = train_test_split(X_train, y_train_oh, test_size=0.15, random_state=42,
                                                          stratify=y_train_oh)

        histories = []
        val_accuracies = []
        all_results = []

        param_combinations = product(*param_grid.values())
        param_names = list(param_grid.keys())

        # grid search
        for params in param_combinations:
            epochs, batch_size, lr, momentum, decay = params

            optimizer_kwargs = {'lr': lr, 'momentum': momentum, 'decay': decay}
            optimizer = SGD(**optimizer_kwargs)

            # init model
            model = MLP(layer_sizes = [X_train.shape[1], 128, y_train_oh.shape[1]],
                        act_fn=ReLU(),
                        loss_fn=CrossEntropyLoss(),
                        softmax_fn=Softmax(),
                        weight_initializer=kaiming)

            # train
            history = model.fit(X_train, y_train_oh, optimizer, X_test=X_val, y_test=y_val_oh, batch_size=batch_size, epochs=epochs, verbose=False)
            histories.append(history)

            # eval on validation set
            _, val_accuracy = model.calculate_metrics(X_val, y_val_oh)
            val_accuracies.append(val_accuracy)

            current_run_data = dict(zip(param_names, params))
            current_run_data['val_accuracy'] = val_accuracy
            all_results.append(current_run_data)

            # Optionally print out the results for each combination of parameters
            if verbose:
                params_dict = dict(zip(param_names, params))
                params_str = ', '.join(f'{key}={value}' for key, value in params_dict.items())
                print(f"Completed training with params: {params_str}. Validation Accuracy: {val_accuracy:.4f}\n")

        # save results
        with open(f'{filepath}/histories.pickle', 'wb') as f:
            pickle.dump(histories, f)
        with open(f'{filepath}/val_accuracies.pickle', 'wb') as f:
            pickle.dump(val_accuracies, f)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'{filepath}/grid_search_results.csv', index=False)

        return histories, val_accuracies

    else:
        results = pd.read_csv(f'{filepath}/grid_search_results.csv')
        return results.sort_values(by=['val_accuracy'], ascending=False)


def regularization_grid_search(param_grid, filepath='./out/grid_search/regularization', verbose=False, RUN_EXP=False):


    if RUN_EXP:
        X_train, X_test, y_train_oh, y_test_oh = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST',
                                                                          normalize=True, mlp=True)
        X_train, X_val, y_train_oh, y_val_oh = train_test_split(X_train, y_train_oh, test_size=0.15, random_state=42,
                                                                stratify=y_train_oh)

        histories = []
        val_accuracies = []
        all_results = []

        param_combinations = product(*param_grid.values())
        param_names = list(param_grid.keys())

        # grid search
        for params in param_combinations:
            epochs, batch_size, lr, momentum, decay, regularization, lambd = params

            optimizer_kwargs = {'lr': lr, 'momentum': momentum, 'decay': decay, 'regularization': regularization, 'lambd': lambd}
            optimizer = SGD(**optimizer_kwargs)

            # init model
            model = MLP(layer_sizes = [X_train.shape[1], 128, 128, y_train_oh.shape[1]],
                        act_fn=ReLU(),
                        loss_fn=CrossEntropyLoss(),
                        softmax_fn=Softmax(),
                        weight_initializer=kaiming)

            # train
            history = model.fit(X_train, y_train_oh, optimizer, X_test=X_val, y_test=y_val_oh, batch_size=batch_size, epochs=epochs, verbose=False)
            histories.append(history)

            # eval on validation set
            _, val_accuracy = model.calculate_metrics(X_val, y_val_oh)
            val_accuracies.append(val_accuracy)

            current_run_data = dict(zip(param_names, params))
            current_run_data['val_accuracy'] = val_accuracy
            all_results.append(current_run_data)

            # Optionally print out the results for each combination of parameters
            if verbose:
                params_dict = dict(zip(param_names, params))
                params_str = ', '.join(f'{key}={value}' for key, value in params_dict.items())
                print(f"Completed training with params: {params_str}. Validation Accuracy: {val_accuracy:.4f}\n")

        # save results
        with open(f'{filepath}/histories.pickle', 'wb') as f:
            pickle.dump(histories, f)
        with open(f'{filepath}/val_accuracies.pickle', 'wb') as f:
            pickle.dump(val_accuracies, f)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'{filepath}/grid_search_results.csv', index=False)

        return histories, val_accuracies

    else:
        results = pd.read_csv(f'{filepath}/grid_search_results.csv')
        return results.sort_values(by=['val_accuracy'], ascending=False)


def cnn_grid_search(param_grid, filepath='./out/grid_search/cnn', verbose=False):

    X_train, X_test, y_train, y_test = load_and_preprocess_data('./data/F_MNIST_data', dataset_name='F_MNIST', normalize=True, mlp=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    valset = torch.utils.data.TensorDataset(X_val, y_val)

    histories = []
    val_accuracies = []
    all_results = []

    param_combinations = product(*param_grid.values())
    param_names = list(param_grid.keys())

    # grid search
    for params in param_combinations:
        conv1_out, conv2_out, stride, kernel_size, padding, optimizer_name, lr, momentum, batch_size, epochs = params
        if optimizer_name == 'SGD':
            optimizer_kwargs = {'lr': lr, 'momentum': momentum}
        elif optimizer_name == 'Adam':
            optimizer_kwargs = {'lr': lr}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

        # init model
        model = CNN(input_channels=X_train.shape[1], image_size=X_train.shape[2], conv1_out=conv1_out, conv2_out=conv2_out, stride=stride,
                    kernel_size=kernel_size, padding=padding, optimizer=optimizer_name, **optimizer_kwargs)

        # train
        history = model.fit(trainloader, epochs=epochs, verbose=True, save_every_n_batches=1000)
        histories.append(history)

        # eval on validation set
        _, val_accuracy = model.evaluate(valloader)
        val_accuracies.append(val_accuracy)

        current_run_data = dict(zip(param_names, params))  
        current_run_data['val_accuracy'] = val_accuracy
        all_results.append(current_run_data)

        # Optionally print out the results for each combination of parameters
        if verbose:
            params_dict = dict(zip(param_names, params))
            params_str = ', '.join(f'{key}={value}' for key, value in params_dict.items())
            print(f"Completed training with params: {params_str}. Validation Accuracy: {val_accuracy:.4f}\n")

    # save results
    with open(f'{filepath}/histories.pickle', 'wb') as f:
        pickle.dump(histories, f)
    with open(f'{filepath}/val_accuracies.pickle', 'wb') as f:
        pickle.dump(val_accuracies, f)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{filepath}/grid_search_results.csv', index=False)
            
    return histories, val_accuracies


if __name__ == '__main__':

    ### Grid Search ###
    ## MLP ##
    param_grid = {
        'epochs': [50],
        'batch_size': [128],
        'lr': [0.01],
        'momentum': [0.95],
        'decay': [1e-7],
        'regularization': ['l1', 'l2'],
        'lambd': [5e-3, 1e-3, 1e-4]
    }
    #mlp_grid_search(param_grid, verbose=True)

    ## Regularization Constants ##
    regularization_grid_search(param_grid, verbose=True)
    exit()

    ## CNN ##
    param_grid = {
        'conv1_out': [16, 32],
        'conv2_out': [32, 64],
        'stride': [1],
        'kernel_size': [3, 5],
        'padding': [1, 2],
        'optimizer': ['SGD'],
        'lr': [0.001, 0.01],
        'momentum': [0.0, 0.5, 0.9],
        'batch_size': [16, 32, 64],
        'epochs': [5]
    }
    #cnn_grid_search(param_grid, verbose=True)

    ## Experiment 1 ##
    optimizer_kwargs = {
        'lr': 0.01, 
        'momentum': 0.9
        }
    optimizer = 'SGD'
    batch_size = 64
    epochs = 50
    #exp1(optimizer_kwargs, optimizer_name=optimizer, epochs=epochs, batch_size=batch_size)

    ## Experiment 2 ##
    optimizer_kwargs = {
        'lr': 0.01, 
        'momentum': 0.9
        }
    optimizer = 'SGD'
    batch_size = 64
    epochs = 25
    #exp2(optimizer_kwargs, optimizer_name=optimizer, epochs=epochs, batch_size=batch_size)
        
    ## Experiment 3 ##
    optimizer_kwargs = {
        'lr': 0.01, 
        'momentum': 0.9
        }
    optimizer = 'SGD'
    batch_size = 64
    epochs = 25
    #exp3(optimizer_kwargs, optimizer_name=optimizer, epochs=epochs, batch_size=batch_size)

    ## Experiment 4 ##
    optimizer_kwargs = {
        'lr': 0.01, 
        'momentum': 0.9,
        'lambd': 0.001
        }
    optimizer = 'SGD'
    batch_size = 64
    epochs = 25
    exp4(optimizer_kwargs, optimizer_name=optimizer, epochs=epochs, batch_size=batch_size)

    ## Experiment 5 ##
    optimizer_kwargs = {
        'lr': 0.01, 
        'momentum': 0.9
        }
    optimizer = 'SGD'
    batch_size = 64
    epochs = 25
    #exp5(optimizer_kwargs, optimizer_name=optimizer, epochs=epochs, batch_size=batch_size)

    ## Experiment 6 ##
    optimizer_kwargs = {
        'lr': 0.01, 
        'momentum': 0.9,
        }
    conv1_out = 32
    conv2_out = 64
    stride = 1
    kernel = 5
    padding = 2
    batch_size = 32
    epochs = 5
    optimizer = 'SGD'
    #exp6(optimizer_kwargs, conv1_out=conv1_out, conv2_out=conv2_out, stride=stride, kernel=kernel, padding=padding, epochs=epochs, batch_size=batch_size, verbose=True)
    
    ## Experiment 7 ##
    params_mlp = {
        'hidden_layer_size': 128,
        'epochs': 25,
        'batch_size': 64,
        'lr': 0.01,
        'momentum': 0.9,
        'optimizer': 'SGD'
    }
    params_cnn = {
        'conv1_out': 32,
        'conv2_out': 64,
        'stride': 1,
        'kernel_size': 5,
        'padding': 2,
        'epochs': 5,
        'batch_size': 32,
        'lr': 0.01,
        'momentum': 0.9,
        'optimizer': 'SGD'
    }
    #exp7(params_mlp, params_cnn, verbose=True)

    ## Experiment 8 ##
    lr_sgd = 0.01
    lr_adam = 0.001
    conv1_out = 32
    conv2_out = 64
    stride = 1
    kernel = 5
    padding = 2
    batch_size = 32
    epochs = 5
    #exp8(lr_sgd=lr_sgd, lr_adam=lr_adam, conv1_out=conv1_out, conv2_out=conv2_out, stride=stride, kernel=kernel, padding=padding, epochs=epochs, batch_size=batch_size, verbose=True)
    