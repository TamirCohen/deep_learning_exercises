# create pytorch script
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple# create pytorch script
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

DATASET_PATH = './data_set'
EPOCH_NUMBER = 10
BATCH_SIZE = 128
LERANING_RATE = 0.07
MOMENTUM = 0.9

########### EXPERMINETS ###########
# LERANING_RATE = 0.05, BATCH_SIZE = 64 was good, but converges slow
# LERANING_RATE = 0.01, BATCH_SIZE = 64 converges fast, but oscillates a lot
# LERANING_RATE = 0.07, MOMENTUM = 0.9, BATCH_SIZE = 128, EPOCH_NUMBER = 10. the increased batch size reduced the oscilations. It converges to 0.9?

#TODO consider adding early stopping

class ModelTrainer():
    LOSS_LOG_INTERVAL = 100
    def __init__(self, model, learning_rate, momentum, batch_size, epoch_number, training_loader, test_loader) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_number = epoch_number
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        self.writer = SummaryWriter('runs/fashion_trainer_Parameters_{}_{}_{}'.format(learning_rate, momentum, batch_size))
        print("Using {} device".format(self.device))
    
    def _train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        training_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)


            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % self.LOSS_LOG_INTERVAL == self.LOSS_LOG_INTERVAL - 1:
                training_loss = running_loss / self.LOSS_LOG_INTERVAL # LOSS_LOG_INTERVAL  per batch
                print(f'  batch {i + 1} loss: {training_loss}')
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', training_loss, tb_x)
                running_loss = 0.

        return training_loss
    
    def train_model(self):
        for epoch in range(1, self.epoch_number):
            self.model.train(True)
            train_loss = self._train_one_epoch(epoch, self.writer)
            
            running_test_loss = 0.0
            for i, data in enumerate(test_loader):
                test_inputs, test_lables = data
                test_inputs, test_lables = test_inputs.to(self.device), test_lables.to(self.device)

                voutputs = self.model(test_inputs)
                vloss = self.loss_fn(voutputs, test_lables)
                running_test_loss += vloss

            test_loss = running_test_loss / (i + 1)
            print(f'Epoch {epoch} LOSS train {train_loss} test {test_loss}')
            # Log the running loss averaged per batch
            # for both training and test
            self.writer.add_scalars('Training vs. test Loss',
                        { 'Training' : train_loss, 'test' : test_loss },
                        epoch + 1)
            test_accuracy = self.calculate_accuracy(self.model, test_loader)
            self.writer.add_scalar('Accuracy/test', test_accuracy, epoch + 1)
            print(f"test accuracy: {test_accuracy}")
            self.writer.flush()
        self.writer.close()

    def calculate_accuracy(self, model, data_loader):
        # Set model to evaluation mode
        model.eval()
        
        # Initialize counters
        num_correct = 0
        num_total = 0
        
        # Iterate over the test dataset
        with torch.no_grad():
            for data, labels in data_loader:
                # Forward pass
                data, labels = data.to(self.device), labels.to(self.device)
                output = model(data)
                _, predictions = torch.max(output, 1)
                
                # Update counters
                num_correct += (predictions == labels).sum().item()
                num_total += labels.size(0)
        
        # Calculate accuracy
        accuracy = num_correct / num_total
        return accuracy


# Create lenet5 model for fashion MNIST dataset
class LeNet5(nn.Module):
    LAST_CONV_OUT_CHANNEL = 16
    def __init__(self):
        super(LeNet5, self).__init__()
        # Added 2 padding to make the output size same as input size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=self.LAST_CONV_OUT_CHANNEL, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(self.LAST_CONV_OUT_CHANNEL * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = tanh(self.conv1(x))
        x = self.pool1(x)
        x = tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))
        x = self.fc3(x)
        return x

def initialize_data_loaders() -> Tuple[DataLoader, DataLoader]: 
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    training_set = datasets.FashionMNIST(DATASET_PATH, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(DATASET_PATH, train=False, download=True, transform=transform)
    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE, shuffle=False)
    
    return training_loader, test_loader

if __name__ == '__main__':
    training_loader, test_loader = initialize_data_loaders()

    ModelTrainer(LeNet5(), LERANING_RATE, MOMENTUM, BATCH_SIZE, EPOCH_NUMBER, training_loader, test_loader).train_model()

DATASET_PATH = './data_set'
EPOCH_NUMBER = 10
BATCH_SIZE = 128
LERANING_RATE = 0.07
MOMENTUM = 0.9

########### EXPERMINETS ###########
# Without normalization
# LERANING_RATE = 0.05, BATCH_SIZE = 64 was good, but converges slow
# LERANING_RATE = 0.01, BATCH_SIZE = 64 converges fast, but oscillates a lot
# LERANING_RATE = 0.07, MOMENTUM = 0.9, BATCH_SIZE = 128, EPOCH_NUMBER = 10. the increased batch size reduced the oscilations. It converges to 0.9?

#TODO consider adding early stopping

class ModelTrainer():
    LOSS_LOG_INTERVAL = 100
    def __init__(self, model, learning_rate, momentum, batch_size, epoch_number, training_loader, test_loader) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_number = epoch_number
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        self.writer = SummaryWriter('runs/fashion_trainer_Parameters_{}_{}_{}'.format(learning_rate, momentum, batch_size))
        print("Using {} device".format(self.device))
    
    def _train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        training_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)


            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % self.LOSS_LOG_INTERVAL == self.LOSS_LOG_INTERVAL - 1:
                training_loss = running_loss / self.LOSS_LOG_INTERVAL # LOSS_LOG_INTERVAL  per batch
                print(f'  batch {i + 1} loss: {training_loss}')
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', training_loss, tb_x)
                running_loss = 0.

        return training_loss
    
    def train_model(self):
        for epoch in range(1, self.epoch_number):
            self.model.train(True)
            train_loss = self._train_one_epoch(epoch, self.writer)
            
            running_test_loss = 0.0
            for i, data in enumerate(test_loader):
                test_inputs, test_lables = data
                test_inputs, test_lables = test_inputs.to(self.device), test_lables.to(self.device)

                voutputs = self.model(test_inputs)
                vloss = self.loss_fn(voutputs, test_lables)
                running_test_loss += vloss

            test_loss = running_test_loss / (i + 1)
            print(f'Epoch {epoch} LOSS train {train_loss} test {test_loss}')
            # Log the running loss averaged per batch
            # for both training and test
            self.writer.add_scalars('Training vs. test Loss',
                        { 'Training' : train_loss, 'test' : test_loss },
                        epoch + 1)
            test_accuracy = self.calculate_accuracy(self.model, test_loader)
            self.writer.add_scalar('Accuracy/test', test_accuracy, epoch + 1)
            print(f"test accuracy: {test_accuracy}")
            self.writer.flush()
        self.writer.close()

    def calculate_accuracy(self, model, data_loader):
        # Set model to evaluation mode
        model.eval()
        
        # Initialize counters
        num_correct = 0
        num_total = 0
        
        # Iterate over the test dataset
        with torch.no_grad():
            for data, labels in data_loader:
                # Forward pass
                data, labels = data.to(self.device), labels.to(self.device)
                output = model(data)
                _, predictions = torch.max(output, 1)
                
                # Update counters
                num_correct += (predictions == labels).sum().item()
                num_total += labels.size(0)
        
        # Calculate accuracy
        accuracy = num_correct / num_total
        return accuracy


# Create lenet5 model for fashion MNIST dataset
class LeNet5(nn.Module):
    LAST_CONV_OUT_CHANNEL = 16
    def __init__(self):
        super(LeNet5, self).__init__()
        # Added 2 padding to make the output size same as input size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=self.LAST_CONV_OUT_CHANNEL, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(self.LAST_CONV_OUT_CHANNEL * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = tanh(self.conv1(x))
        x = self.pool1(x)
        x = tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))
        x = self.fc3(x)
        return x

def initialize_data_loaders() -> Tuple[DataLoader, DataLoader]: 
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    training_set = datasets.FashionMNIST(DATASET_PATH, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(DATASET_PATH, train=False, download=True, transform=transform)
    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE, shuffle=False)
    
    return training_loader, test_loader

if __name__ == '__main__':
    training_loader, test_loader = initialize_data_loaders()

    ModelTrainer(LeNet5(), LERANING_RATE, MOMENTUM, BATCH_SIZE, EPOCH_NUMBER, training_loader, test_loader).train_model()