import matplotlib.pyplot as plt

def plot_training_history(history, figsize=(12, 4), axes=None, title='', 
                          filename=None, show=False):
    """Plots the training history"""

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1, ax2 = axes  # unpack the axes

    train_loss = history['train_loss']
    test_loss = history['test_loss']
    train_acc = history['train_acc']
    test_acc = history['test_acc']

    # plot train and test loss
    ax1.plot(train_loss, label='Train')
    ax1.plot(test_loss, label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss per Sample')
    ax1.set_title('Loss over Epochs' + ('' if title == '' else f' ({title})'))
    ax1.legend()

    # plot train and test accuracy
    ax2.plot(train_acc, label='Train')
    ax2.plot(test_acc, label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs' + ('' if title == '' else f' ({title})'))
    ax2.legend()

    if filename:
        plt.savefig(filename, dpi=400)
    if show:
        plt.show()


def compare_training_histories(histories, titles, filename=None, show=True):
    """Compares multiple training histories by plotting them in a single figure with multiple subplots."""

    num_histories = len(histories)
    fig, axes = plt.subplots(num_histories, 2, figsize=(12, 4 * num_histories)) 

    for i, history in enumerate(histories):
        plot_training_history(history, axes=axes[i], title=titles[i])

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
