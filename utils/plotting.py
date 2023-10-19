import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, num_saves_per_epoch, figsize=(12, 4), axes=None, title='', 
                          filename=None, show=False):
    """Plots the training history"""

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1, ax2 = axes  # unpack the axes

    train_loss = history['train_loss']
    test_loss = history['test_loss']
    train_acc = history['train_acc']
    test_acc = history['test_acc']

    test = len(train_loss)

    # calc the x-axis values in terms of epochs
    num_epochs = len(train_loss) / num_saves_per_epoch
    x_axis_values = np.linspace(0, num_epochs, len(train_loss))

    # plot train and test loss
    ax1.plot(x_axis_values, train_loss, label='Train')
    if test_loss:
        ax1.plot(x_axis_values, test_loss, label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss per Sample')
    ax1.set_title('Loss over Epochs' + ('' if title == '' else f' ({title})'))
    #ax1.set_xticks(np.arange(1, num_epochs + 1))  # Set custom ticks on the x-axis

    ax1.legend()

    # plot train and test accuracy
    ax2.plot(x_axis_values, train_acc, label='Train')
    if test_acc:
        ax2.plot(x_axis_values, test_acc, label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs' + ('' if title == '' else f' ({title})'))
    ax2.legend()

    if filename:
        plt.savefig(filename, dpi=400)
    if show:
        plt.show()


def compare_training_histories(histories, titles, filename=None, show=True, num_saves_per_epoch=1):
    """Compares multiple training histories by plotting them in a single figure with multiple subplots."""

    num_histories = len(histories)
    fig, axes = plt.subplots(num_histories, 2, figsize=(12, 4 * num_histories)) 

    for i, history in enumerate(histories):
        plot_training_history(history, axes=axes[i], title=titles[i], 
                              num_saves_per_epoch=num_saves_per_epoch)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=400)
    if show:
        plt.show()


def compare_accuracies(histories, labels, figsize=(8, 4), plot_train=True, 
                       title='', filename=None, show=True):
    """Compares the accuracies from multiple training histories on a single plot."""

    plt.figure(figsize=figsize)

    for history, label in zip(histories, labels):
        if plot_train:
            plt.plot(history['train_acc'], label=f'Train ({label})')
            plt.plot(history['test_acc'], label=f'Test ({label})')
        else:
            plt.plot(history['test_acc'], label=label)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=400)
    if show:
        plt.show()
