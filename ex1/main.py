# create pytorch script
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
DATASET_PATH = './data_set'

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Load fashion MNIST dataset 
training_set = datasets.FashionMNIST(DATASET_PATH, train=True, download=True, transform=transform)
validation_set = datasets.FashionMNIST(DATASET_PATH, train=False, download=True, transform=transform)

#TODO remove the partial data set
partial_training_set = Subset(training_set, range(10000))
partial_validation_set = Subset(validation_set, range(1000))


training_loader = torch.utils.data.DataLoader(
    partial_training_set,
    batch_size=10, shuffle=True)

validation_loader = torch.utils.data.DataLoader(
    partial_validation_set,
    batch_size=10, shuffle=False)


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
        x = nn.functional.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.sigmoid(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

# define lenet5 loss function and optimizer
model = LeNet5()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
writer = SummaryWriter()


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

if __name__ == '__main__':
    losses = []
    for epoch in range(1, 20):
        loss = train_one_epoch(epoch, writer)
        losses.append(loss)
        print(f"Epoch {epoch} loss: {loss}")
    writer.flush()
    writer.close()
