# load fashion mnist dataset
# using pytorch

#references:
#https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
#https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Params
BATCH_SIZE = 64
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5
DIM = 128 
NOISE_SIZE = 128

#consts
MODE = 'wgan-gp'

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(NOISE_SIZE, 4*4*4*DIM)
        self.batchnorm1 = nn.BatchNorm2d(4*DIM)
        self.deconv1 = nn.ConvTranspose2d(4*DIM, 2*DIM, 5)
        self.batchnorm2 = nn.BatchNorm2d(2*DIM)
        self.deconv2 = nn.ConvTranspose2d(2*DIM, DIM, 5)
        self.batchnorm3 = nn.BatchNorm2d(DIM)
        self.deconv3 = nn.ConvTranspose2d(DIM, 3, 5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, noise):
        # shape: (batch_size, 128)
        x = self.linear(noise)
        # shape: (batch_size, 4*4*4*DIM)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x.view(-1, 4*DIM, 4, 4)
        # shape: (batch_size, 4*DIM, 4, 4)

        x = self.deconv1(x)
        # shape: (batch_size, 2*DIM, 8, 8)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.deconv2(x)





def load_fashion_mnist():
    # load fashion mnist dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((NORMALIZE_MEAN,), (NORMALIZE_STD,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
    testset = torchvision.datasets.FashionMNIST(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False)
    return trainloader, testloader

def main():
    print("load fashion mnist dataset")
    trainloader, testloader = load_fashion_mnist()
    print("load fashion mnist dataset done")
    # get some random training images
    dataiter = iter(trainloader)
    # get random examples from the training set
    images, labels = next(dataiter)
    # show images
    img_show(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % labels[j] for j in range(64)))

def img_show(img):
    print("showing an image")
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
if __name__ == "__main__":
    main()