# load fashion mnist dataset
# using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

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