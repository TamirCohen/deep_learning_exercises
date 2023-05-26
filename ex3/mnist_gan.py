# load fashion mnist dataset
# using pytorch

#references:
#https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
#https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py

# Reference:
# pytorch implementation https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
# The original CIFAR model

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
EPOCHS = 10
# From PAPER
BATCH_SIZE = 64
# consider changing this to 64
MODEL_DIMENSION = 128 
NOISE_SIZE = 128
LEAKY_SLOPE = 0.2
STRIDE = 2
KERNEL_SIZE = 4
DCGAN_BETA1 = 0.5
DCGAN_BETA2 = 0.999
DCGAN_LEARNING_RATE = 0.0002
#consts
MODE = 'dcgan'
IMAGE_DIM = 28
OUTPUT_DIM = IMAGE_DIM ** 2
DISCRIMINATOR_ITERATIONS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#TODO consider increasing the MODEL_DIMENSION throughout the model the 4 * 4 * 4
class Discriminator(nn.Module):
    def __init__(self, model_type) -> None:
        super().__init__()
        layers = [nn.Conv2d(1, MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Conv2d(MODEL_DIMENSION, 2 * MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE)]
        if model_type != 'wgan-gp':
            layers.append(nn.BatchNorm2d(2 * MODEL_DIMENSION))
        layers += [nn.LeakyReLU(LEAKY_SLOPE),
            nn.Conv2d(2 * MODEL_DIMENSION, 4 * MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE)]
        if model_type != 'wgan-gp':
            layers.append(nn.BatchNorm2d(4 * MODEL_DIMENSION))
        layers += [nn.LeakyReLU(LEAKY_SLOPE), nn.Flatten(), nn.Linear(4 * MODEL_DIMENSION, 1)]
        # Adding sigmoid here because I want to use BCELoss in DC-GAN
        if model_type == 'dcgan':
            layers.append(nn.Sigmoid())
        
        self.module = nn.Sequential(*layers)
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
            nn.Linear(NOISE_SIZE, 3 * 3 * 4 * MODEL_DIMENSION),
            nn.BatchNorm1d(3 * 3 * 4 * MODEL_DIMENSION),
            nn.ReLU(),
        )

        self.dconv_upsample = nn.Sequential(
            nn.ConvTranspose2d(4 * MODEL_DIMENSION, 2 * MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
            nn.BatchNorm2d(2 * MODEL_DIMENSION),
            nn.ReLU()
        )
        self.dconv_upsample2 = nn.Sequential(
            nn.ConvTranspose2d(2 * MODEL_DIMENSION, MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(MODEL_DIMENSION),
            nn.ReLU()
        )

        #TODO this is not like the paper - I increased the kernel size so it will match exactly the image size
        self.dconv_upsample3 = nn.Sequential(
            nn.ConvTranspose2d(MODEL_DIMENSION, 1, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise):
        output = self.upsample_noise(noise)
        output = output.view(BATCH_SIZE, 4 * MODEL_DIMENSION, 3, 3)
        output = self.dconv_upsample(output)
        output = self.dconv_upsample2(output)
        output = self.dconv_upsample3(output)
        return output.view(BATCH_SIZE, OUTPUT_DIM)

def load_fashion_mnist():
    # load fashion mnist dataset
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((NORMALIZE_MEAN,), (NORMALIZE_STD,))])
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((NORMALIZE_MEAN,), (NORMALIZE_STD,))])
    
    trainset = torchvision.datasets.FashionMNIST(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
    testset = torchvision.datasets.FashionMNIST(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transforms.ToTensor())
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

def get_optimizer(discriminator, generator, mode):
    if mode == "dcgan":
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=DCGAN_LEARNING_RATE, betas=(DCGAN_BETA1, DCGAN_BETA2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=DCGAN_LEARNING_RATE, betas=(DCGAN_BETA1, DCGAN_BETA2))
    elif mode == "wgan-gp":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return optimizer_G, optimizer_D

def train_discriminator(discriminator, generator, optimizer_D, real_images, labels, mode):
    binary_cross_entropy_loss =  nn.BCELoss()
    noise = torch.randn(BATCH_SIZE, NOISE_SIZE).to(DEVICE)
    fake_images = generator(noise)
    discriminator_fake = discriminator(fake_images)
    discriminator_real = discriminator(real_images)
    if mode == "dcgan":
        discriminator_loss = binary_cross_entropy_loss(discriminator_fake, torch.zeros_like(discriminator_fake))
        discriminator_loss += binary_cross_entropy_loss(discriminator_real, torch.ones_like(discriminator_real))
        discriminator_loss /= 2
        discriminator.zero_grad()
        discriminator_loss.backward()
    elif mode == "wgan-gp":
        # Minimizing loss according to Kantorovich-Rubinstein duality to the Wasserstein distance
        discriminator_loss_real = discriminator_real.mean(0).view(1)
        discriminator_loss_real.backward()

        discriminator_loss_fake = -discriminator_fake.mean(0).view(1)
        discriminator_loss_fake.backward()
        discriminator_loss = discriminator_loss_real + discriminator_loss_fake
        # TODO: Implement gradient penalty :)
        raise NotImplementedError
    else:
        raise NotImplementedError
    optimizer_D.step()
    return discriminator_loss

def train_generator(discriminator, generator, optimizer_G, mode):
    binary_cross_entropy_loss =  nn.BCELoss()
    noise = torch.randn(BATCH_SIZE, NOISE_SIZE).to(DEVICE)
    fake_images = generator(noise)
    discriminator_fake = discriminator(fake_images)
    if mode == "dcgan":
        generator_loss = binary_cross_entropy_loss(discriminator_fake, torch.ones_like(discriminator_fake))
    else:
        raise NotImplementedError
    generator.zero_grad()
    discriminator.zero_grad()
    generator_loss.backward()
    optimizer_G.step()
    return generator_loss

def train(trainloader, discriminator, generator, optimizer_G, optimizer_D, mode):
    for epoch in range(EPOCHS):
        for iteration, (real_images, labels) in enumerate(trainloader):
            if len(real_images) != BATCH_SIZE:
                continue
            # Training the Discriminator
            real_images = real_images.to(DEVICE)
            labels = labels.to(DEVICE)
            discriminator_loss = train_discriminator(discriminator, generator, optimizer_D, real_images, labels, mode)
            # Training the Generator
            if mode == "dcgan":
                discriminator_iterations = 1
            else:
                discriminator_iterations = DISCRIMINATOR_ITERATIONS

            if iteration % discriminator_iterations == 0:
               generator_loss = train_generator(discriminator, generator, optimizer_G, mode)
               print("Iteration: {} / {}, GenLoss: {} , DiscLoss: {}".format(iteration + 1, len(trainloader), generator_loss.item(), discriminator_loss.item()))

def main():
    print("load fashion mnist dataset")
    trainloader, testloader = load_fashion_mnist()
    print("load fashion mnist dataset done")
    generator = Generator().to(DEVICE)
    discriminator = Discriminator(MODE).to(DEVICE)
    # sample = next(iter(trainloader))[0]
    # import pdb; pdb.set_trace()
    print("Generator Netowrk")
    print(generator)

    print("Discrimnator Netowrk")
    print(discriminator)
    optimizer_G, optimizer_D = get_optimizer(discriminator, generator, MODE)
    train(trainloader, discriminator, generator, optimizer_G, optimizer_D, MODE)

    # # get some random training images
    # dataiter = iter(trainloader)
    # # # get random examples from the training set
    # images, labels = next(dataiter)
    # # # show images
    # # img_show(torchvision.utils.make_grid(images))
    # # # print labels
    # # print(' '.join('%5s' % labels[j] for j in range(64)))


if __name__ == "__main__":
    main()