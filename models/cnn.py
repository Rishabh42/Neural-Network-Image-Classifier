import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CNN(nn.Module):
    def __init__(self, input_channels, image_size, conv1_out, conv2_out, kernel_size, 
                 stride, padding, num_classes=10, optimizer='SGD', **optimizer_kwargs):  
              
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, conv1_out, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size, stride=stride, padding=padding)

        size = (image_size - kernel_size + 2*padding) // stride + 1
        size = size // 2  # max pooling
        size = (size - kernel_size + 2*padding) // stride + 1
        size = size // 2

        self.fc1 = nn.Linear(conv2_out * size * size, 128)
        self.fc2 = nn.Linear(128, num_classes) 

        self.pool = nn.MaxPool2d(2, 2)

        self.criterion = nn.CrossEntropyLoss()
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), **optimizer_kwargs)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), **optimizer_kwargs)


    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dims except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def fit(self, train_loader, test_loader=None, epochs=5, print_every=1, verbose=True, track_history=True):
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        for epoch in range(epochs):
            self.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for _, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self(data)
                loss = self.criterion(output, target.type(torch.LongTensor))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)  # get the idx of the max log-prob
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            if track_history:
                train_loss = running_loss / len(train_loader.dataset)
                train_acc = correct / total
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)

                if test_loader is not None:
                    test_loss, test_acc = self.evaluate(test_loader)
                    history['test_loss'].append(test_loss)
                    history['test_acc'].append(test_acc)

            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                if test_loader is not None:
                    print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}')
                else:
                    print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}')

        return history

    def evaluate(self, test_loader):
        self.eval() # set the model to eval mode
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self(data)
                test_loss += self.criterion(output, target.type(torch.LongTensor)).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        test_loss /= len(test_loader.dataset)
        test_acc = correct / total

        return test_loss, test_acc
