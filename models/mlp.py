import numpy as np

class MLP:
    def __init__(self, layer_sizes, act_fn, loss_fn, softmax_fn, weight_initializer):
        self.layer_sizes = layer_sizes
        self.weight_initializer = weight_initializer
        self.act_fn = act_fn
        self.loss_fn = loss_fn
        self.softmax = softmax_fn

        # init weights and biases
        self.w = [self.weight_initializer((layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        self.b = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

    def forward(self, X):
        a = [X] # input layer
        
        for i in range(len(self.w)-1): # hidden layers
            z = a[-1] @ self.w[i] + self.b[i]
            a.append(self.act_fn.forward(z))
        
        z = a[-1] @ self.w[-1] + self.b[-1] # output layer
        a.append(self.softmax.forward(z))
        return a

    def backward(self, X, y, a):    
        m = X.shape[0] # for averaging
        grads = {}
        output_error = self.loss_fn.gradient(y, a[-1])
        da = output_error

        for i in reversed(range(len(self.w))):
            # hidden: product of the error of next layer (da) and the deriv of act_fn wrt z
            dz = da * self.act_fn.gradient(a[i+1]) if i < len(self.w)-1 else da # da for output layer
            grads[f'W{i+1}'] = a[i].T @ dz / m
            grads[f'b{i+1}'] = np.sum(dz, axis=0, keepdims=True) / m
            da = dz @ self.w[i].T # derivative of loss wrt a[i]

        return grads

    def fit(self, X, y, optimizer, X_test=None, y_test=None, epochs=100, batch_size=64, 
            print_every=10, verbose=True, track_history=True):
        num_samples = X.shape[0]
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        for epoch in range(epochs):
            permuted_indices = np.random.permutation(num_samples)
            X_shuffled = X[permuted_indices]
            y_shuffled = y[permuted_indices]

            batch_losses, batch_accuracies = [], []
            for i in range(0, num_samples, batch_size):
                X_batch, y_batch = X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]

                # fwd pass, bwd pass, update step
                a = self.forward(X_batch)
                grads = self.backward(X_batch, y_batch, a)
                optimizer.update(self.w, self.b, grads)

                if track_history:
                    batch_loss, batch_acc = self.calculate_metrics(X_batch, y_batch)
                    batch_losses.append(batch_loss)
                    batch_accuracies.append(batch_acc)

            if track_history:
                train_loss, train_acc = np.mean(batch_losses), np.mean(batch_accuracies)
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)

                if X_test is not None and y_test is not None:
                    test_loss, test_acc = self.calculate_metrics(X_test, y_test)
                    history['test_loss'].append(test_loss)
                    history['test_acc'].append(test_acc)

            if verbose and epoch % print_every == 0:
                if X_test is not None and y_test is not None:
                    print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}')
                else:
                    print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}')

        return history
    
    def calculate_metrics(self, X, y):
        a = self.forward(X)
        loss = self.loss_fn.loss(y, a[-1]) / X.shape[0]
        acc = np.mean(np.argmax(a[-1], axis=1) == np.argmax(y, axis=1))
        return loss, acc

    def predict(self, X):
        a = self.forward(X)
        y_preds = np.argmax(a[-1], axis=1)
        return y_preds