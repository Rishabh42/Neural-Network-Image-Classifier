import matplotlib.pyplot as plt

def plot_training_history(history, figsize=(12, 4), axes=None, title='', 
                          filename=None, show=True):
    """
    Plots the training history.

    Parameters:
    - history: dict containing the training history (loss and accuracy for both training and test sets).
    - axes: list of matplotlib axes objects where the plots will be drawn.
    - title: str, optional title to give to the subplot.
    """

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1, ax2 = axes  # unpack the axes

    train_loss = history['train_loss']
    test_loss = history['test_loss']
    train_acc = history['train_acc']
    test_acc = history['test_acc']

    # plot train and test loss
    ax1.plot(train_loss, label='Train Loss')
    ax1.plot(test_loss, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss per Sample')
    ax1.set_title('Loss over Epochs' + ('' if title == '' else f' ({title})'))
    ax1.legend()

    # plot train and test accuracy
    ax2.plot(train_acc, label='Train Acc')
    ax2.plot(test_acc, label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs' + ('' if title == '' else f' ({title})'))
    ax2.legend()

    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.close()


def compare_training_histories(histories, titles, filename=None, show=True):
    """
    Compares multiple training histories by plotting them in a single figure with multiple subplots.

    Parameters:
    - histories: list of dicts, each containing a training history.
    - titles: list of strings, titles for each of the subplots.
    """

    num_histories = len(histories)
    fig, axes = plt.subplots(num_histories, 2, figsize=(12, 4 * num_histories)) 

    for i, history in enumerate(histories):
        plot_training_history(history, axes=axes[i], title=titles[i])

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.close()


def compare_accuracies(histories, labels, figsize=(8, 4), plot_train=True, 
                       title='', filename=None, show=True):
    """
    Compares the accuracies from multiple training histories on a single plot.

    Parameters:
    - histories: list of dicts, each containing a training history.
    - labels: list of strings, labels for each accuracy plot.
    - plot_title: string, title for the plot.
    """

    plt.figure(figsize=figsize)

    for history, label in zip(histories, labels):
        if plot_train:
            plt.plot(history['train_acc'], label=f'Train Acc ({label})')
        plt.plot(history['test_acc'], label=f'Test Acc ({label})')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.close()

