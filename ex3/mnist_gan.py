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

# I invented these
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

# From PAPER
BATCH_SIZE = 64
# consider changing this to 64
MODEL_DIMENSION = 128 
NOISE_SIZE = 128
LEAKY_SLOPE = 0.2
STRIDE = 2
KERNEL_SIZE = 4

#consts
MODE = 'wgan-gp'
IMAGE_DIM = 28
OUTPUT_DIM = IMAGE_DIM ** 2

#TODO consider increasing the MODEL_DIMENSION throughout the model the 4 * 4 * 4
class Discriminator(nn.Module):
    def __init__(self, model_type) -> None:
        super().__init__()
        assert model_type in ['wgan-gp']
        self.module = nn.Sequential(
            nn.Conv2d(1, MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Conv2d(MODEL_DIMENSION, MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Conv2d(MODEL_DIMENSION, MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Flatten(),
            nn.Linear(MODEL_DIMENSION, 1)
        )
        self.model_type = model_type
    
    def forward(self, x):
        # TODO change view to be for fashion mnist
        x = x.view(BATCH_SIZE, 1, IMAGE_DIM, IMAGE_DIM)
        return self.module(x)
# MODELS
class Generator(nn.Module):
    """
    The original Generator was needed to generate pictures of 3*32*32
    I changed it to generate pictures of 1*28*28
    the input of this model layers are:
    torch.Size([64, 128])
    torch.Size([64, 1152])
    torch.Size([64, 128, 3, 3])
    torch.Size([64, 128, 6, 6])
    torch.Size([64, 128, 14, 14])
    torch.Size([64, 1, 28, 28])
    """
    def __init__(self):
        super(Generator, self).__init__()

        # Using 3 instead of 4 because I want to get to 24 and than to 28
        self.upsample_noise = nn.Sequential(
            nn.Linear(NOISE_SIZE, 3 * 3 * MODEL_DIMENSION),
            nn.BatchNorm1d(3 * 3 * MODEL_DIMENSION),
            nn.ReLU(),
        )

        self.dconv_upsample = nn.Sequential(
            nn.ConvTranspose2d(MODEL_DIMENSION, MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
            nn.BatchNorm2d(MODEL_DIMENSION),
            nn.ReLU()
        )
        self.dconv_upsample2 = nn.Sequential(
            nn.ConvTranspose2d(MODEL_DIMENSION, MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(MODEL_DIMENSION),
            nn.ReLU()
        )

        #TODO this is not like the paper - I increased the kernel size so it will match exactly the image size
        self.dconv_upsample3 = nn.Sequential(
            nn.ConvTranspose2d(MODEL_DIMENSION, 1, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise):
        print(noise.shape)
        output = self.upsample_noise(noise)
        print(output.shape)
        output = output.view(BATCH_SIZE, MODEL_DIMENSION, 3, 3)
        print(output.shape)
        output = self.dconv_upsample(output)
        print(output.shape)
        output = self.dconv_upsample2(output)
        print(output.shape)
        output = self.dconv_upsample3(output)
        print(output.shape)
        return output.view(BATCH_SIZE, OUTPUT_DIM)

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

def img_show(img):
    print("showing an image")
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


def main():
    print("load fashion mnist dataset")
    trainloader, testloader = load_fashion_mnist()
    print("load fashion mnist dataset done")
    model = Generator()
    gen_output = model(torch.randn(BATCH_SIZE, NOISE_SIZE))
    discriminator = Discriminator(MODE)
    print(discriminator(gen_output))
    # get some random training images
    dataiter = iter(trainloader)
    # # get random examples from the training set
    images, labels = next(dataiter)
    # # show images
    # img_show(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % labels[j] for j in range(64)))


if __name__ == "__main__":
    main()