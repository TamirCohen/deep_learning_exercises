# load fashion mnist dataset
# using pytorch

#references:
#https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
# I invented these
EPOCHS = 50
WGAN_EPOCHS = 30
# From PAPER
BATCH_SIZE = 64
# MODEL_DIMENSION was 128 in the paper but they suggested to use 64 to prevent overfitting
MODEL_DIMENSION = 64
NOISE_SIZE = 128
LEAKY_SLOPE = 0.2
STRIDE = 2
KERNEL_SIZE = 4
DCGAN_BETA1 = 0.5
DCGAN_BETA2 = 0.999
DCGAN_LEARNING_RATE = 0.0002
WGAN_LEARNING_RATE = 0.00005
WGAN_WEIGHT_CLIP = 0.01

#consts
IMAGE_DIM = 28
OUTPUT_DIM = IMAGE_DIM ** 2
DISCRIMINATOR_ITERATIONS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 100
#TODO consider increasing the MODEL_DIMENSION throughout the model the 4 * 4 * 4
class Discriminator(nn.Module):
    def __init__(self, model_type) -> None:
        super().__init__()
        layers = [nn.Conv2d(1, MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
            nn.LeakyReLU(LEAKY_SLOPE, inplace=True),
            nn.Conv2d(MODEL_DIMENSION, 2 * MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)]
        if model_type != 'wgan-gp':
            layers.append(nn.BatchNorm2d(2 * MODEL_DIMENSION))
        layers += [nn.LeakyReLU(LEAKY_SLOPE, inplace=True),
            nn.Conv2d(2 * MODEL_DIMENSION, 4 * MODEL_DIMENSION, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)]
        if model_type != 'wgan-gp':
            layers.append(nn.BatchNorm2d(4 * MODEL_DIMENSION))
        layers += [nn.LeakyReLU(LEAKY_SLOPE, inplace=True), nn.Flatten(), nn.Linear(3 * 3 * 4 * MODEL_DIMENSION, 1)]
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
        output = output.view(-1, 4 * MODEL_DIMENSION, 3, 3)
        output = self.dconv_upsample(output)
        output = self.dconv_upsample2(output)
        output = self.dconv_upsample3(output)
        return output.view(-1, OUTPUT_DIM)

def load_fashion_mnist():
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

def get_optimizer(discriminator, generator, mode):
    if mode == "dcgan":
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=DCGAN_LEARNING_RATE, betas=(DCGAN_BETA1, DCGAN_BETA2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=DCGAN_LEARNING_RATE, betas=(DCGAN_BETA1, DCGAN_BETA2))
    elif mode == "wgan-gp":
        raise NotImplementedError
    elif mode == "wgan":
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=WGAN_LEARNING_RATE)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=WGAN_LEARNING_RATE)
    else:
        raise NotImplementedError
    return optimizer_G, optimizer_D

def train_discriminator(discriminator, generator, optimizer_D, real_images, labels, mode):
    binary_cross_entropy_loss =  nn.BCELoss()
    noise = torch.randn(BATCH_SIZE, NOISE_SIZE).to(DEVICE)
    fake_images = generator(noise)
    discriminator_fake = discriminator(fake_images)
    discriminator_real = discriminator(real_images)
    discriminator.zero_grad()
    if mode == "dcgan":
        discriminator_loss = binary_cross_entropy_loss(discriminator_fake, torch.zeros_like(discriminator_fake))
        discriminator_loss += binary_cross_entropy_loss(discriminator_real, torch.ones_like(discriminator_real))
        discriminator_loss.backward()
    elif mode == "wgan-gp" or mode == "wgan":
        # Minimizing loss according to Kantorovich-Rubinstein duality to the Wasserstein distance
        #TODO validate the signs
        discriminator_loss = discriminator_fake.mean(0).view(1) + -discriminator_real.mean(0).view(1)
        discriminator_loss.backward()
        if mode == "wgan-gp":
            raise NotImplementedError
        else:
            for p in discriminator.parameters():
                p.data.clamp_(-WGAN_WEIGHT_CLIP, WGAN_WEIGHT_CLIP)
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
    elif mode == "wgan-gp" or mode == "wgan":
        #TODO validate sign
        generator_loss = -discriminator_fake.mean(0).view(1)
    else:
        raise NotImplementedError
    generator.zero_grad()
    discriminator.zero_grad()
    generator_loss.backward()
    optimizer_G.step()
    return generator_loss

def train(trainloader, discriminator, generator, optimizer_G, optimizer_D, mode, epochs):
    writer = SummaryWriter()
    try:
        for epoch in range(epochs):
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
                if iteration % LOG_INTERVAL == 0:
                    print("Epoch {} / {}, Iteration: {} / {}, GenLoss: {} , DiscLoss: {}".format(epoch, EPOCHS, iteration + 1, len(trainloader), generator_loss.item(), discriminator_loss.item()))
                    writer.add_scalars('loss', {"Genrator Loss": generator_loss.item(), "Discriminator Loss": discriminator_loss.item()}, epoch * len(trainloader) + iteration)
    finally:    
        display_fake_images(generator)
        #TODO fix it 
        torch.save(generator.state_dict(), "generator_{}.pt".format(mode))

def display_images(images, name):
    grid = torchvision.utils.make_grid(images)
    writer = SummaryWriter()
    writer.add_image(name, grid, 0)

def display_fake_images(generator, name="fake_images", image_number=BATCH_SIZE):
    noise = torch.randn(image_number, NOISE_SIZE).to(DEVICE)
    fake_images = generator(noise)
    fake_images = fake_images.view(image_number, 1, IMAGE_DIM, IMAGE_DIM)
    display_images(fake_images, name)

# add argparse to main, should be able to choose between dcgan, wgan, wgan-gp and should train or or generate image from pretrained model
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="dcgan, wgan, wgan-gp")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--generate", action="store_true", help="generate images from pretrained model")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("load fashion mnist dataset")
    trainloader, testloader = load_fashion_mnist()
    print("load fashion mnist dataset done")
    
    if args.train:
        generator = Generator().to(DEVICE)
        discriminator = Discriminator(args.mode).to(DEVICE)
        # get random images from the trainloader and show them on tensorboard
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        display_images(images, "real_images")

        print("Generator Netowrk")
        print(generator)
        print("Discrimnator Netowrk")
        print(discriminator)
        optimizer_G, optimizer_D = get_optimizer(discriminator, generator, args.mode)
        if args.mode == "dcgan":
            epochs = EPOCHS
        else:
            epochs = WGAN_EPOCHS
        train(trainloader, discriminator, generator, optimizer_G, optimizer_D, args.mode, epochs)
    elif args.generate:
        generator = Generator().to(DEVICE)
        generator.load_state_dict(torch.load("generator_{}.pt".format(args.mode)))
        generator.eval()
        display_fake_images(generator, image_number=1, name=f"generated_images_{args.mode}")

if __name__ == "__main__":
    main()
