import tensorboard
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import random
import pickle as pickle


# choose n_labels random labels from the labels vector
random.seed(42) # set the seed for (same seed for each run option)
torch.manual_seed(42) # set the seed for (same seed for each run option)

Size = 28 # size of image (28x28)
Input_size = Size*Size # size of image
input_dim = (Size,Size) # dimension of image
N_minibatch = 100 # size of minibatch
N_labels = 20 # number of classes
Learning_rate = 0.0001 # learning rate
Epochs = 30 # number of epochs  
N_hidden = 600 # number of hidden units
N_z = 50 # number of latent variables
Load_model_from_file = False # load model from file
Save_model_to_file = True # load model from file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))
# a semi-supervised learning via a variational autoencoder
# we implement the model in the paper: Semi-Supervised Learning with Deep Generative Models
# Implementation of the M1 model in the paper as describes in algorithm 1 
# It is based on a VAE for feature extraction and a (transductive) SVM for classification
# Implement the suggested for MNIST, and apply it on fashion MNIST dataset.
# Present results for 100, 600, 1000 and 3000 labels, as they present in Table 1 in the paper.

# Main function
def load_data_fashion():
    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root = 'FashionMNIST/',
        train = True,
        download = True,
        transform=ToTensor()
    )

    test_set = torchvision.datasets.FashionMNIST(
        root = 'FashionMNIST/',
        train = False,
        download = True,
        transform=ToTensor()
    )

    # train dataloaders
    train_loader = DataLoader(dataset=train_set, batch_size=N_minibatch, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=N_minibatch, shuffle=True)

    print(train_set.data[0].shape)
    print(len(train_set))
    print(len(test_set))

    return train_loader, test_loader
    
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(Input_size, N_hidden),
            nn.Softplus(),
            nn.Linear(N_hidden, N_hidden),
            nn.Softplus(),
            nn.Linear(N_hidden, N_z*2)
        )
        self.decode = nn.Sequential(
            nn.Linear(N_z, N_hidden),
            nn.Softplus(),
            nn.Linear(N_hidden, N_hidden),
            nn.Softplus(),
            nn.Linear(N_hidden, Input_size),
            nn.Sigmoid()
        )


    def reparameterize_trick(self, mu, log_var):
        # reparameterization trickq
        std = torch.exp(0.5*log_var) # std = e^(1/2 * log_var)
        eps = torch.randn_like(std) # `randn_like` as we need the same dimension as std
        sample = mu + (eps * std) # Sample from random normal distribution with standard deviation `std` and mean `mu`

        return sample
 
    def forward(self, x): # x is the input
        
        # encoding
        x_out = self.encode(x)
        
        # get `mu` and `log_var`
        mu = x_out[:, :N_z] # the first N_z values as mean
        log_var = x_out[:, N_z:] # the other N_z values as variance
        
        # get the latent vector through reparameterization
        z_sampled = self.reparameterize_trick(mu, log_var)
        
        # decoding
        x_reconstruction = self.decode(z_sampled)
        
        return x_reconstruction, mu, log_var, z_sampled


def loss_function(recon_x, x, mu, log_var):
    BCE_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') # reconstruction loss
    KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL divergence loss
    Total_loss = BCE_loss + KLD_loss # total loss
    return Total_loss


def train_model_vae(model, optimizer, data):
    # train the variational autoencoder model
    model.train() # set the model to training mode
    for epoch in range(Epochs):
        train_loss = 0 # initialize the training loss
        optimizer.zero_grad() # clear the gradients
        for i, (tensor_images, labels) in enumerate(data):
            #tensor_images = tensor_images.to(device)
            tensor_images = tensor_images.view(tensor_images.size(0), -1) # reshape the images to vectors
            
            reconstruction, mu, log_var, z_sampled = model(tensor_images) # forward pass
            optimizer.zero_grad() # clear the gradients
            # calculate loss
            loss = loss_function(reconstruction, tensor_images, mu, log_var)
            # backpropagation
            loss.backward() # compute the gradients
            train_loss += loss.item() # accumulate the loss
            optimizer.step() # update the parameters
        train_loss /= len(data.dataset) # compute the average loss

        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss)) # print the training loss per epoch
    return loss, model


def test_model(model, data):
    # test the model on the test set
    model.eval() # set the model to evaluation mode
    reconstructions = torch.empty(0, Input_size)
    z_sampled_mat = torch.empty(0, N_z)
    labels_vec = torch.empty(0)
    mu_mat = torch.empty(0, N_z)
    with torch.no_grad(): # we don't need gradients for the testing phase
        for _, (tensor_images, labels) in enumerate(data): # iterate over batches of test images
            #data = data.to(device)
            tensor_images = tensor_images.view(tensor_images.size(0), -1) # reshape the images to vectors
            reconstruction, mu, logvar, z_sampled = model(tensor_images) # forward pass
            reconstructions = torch.cat((reconstructions, reconstruction), 0) # concatenate the reconstructions
            z_sampled_mat = torch.cat((z_sampled_mat, z_sampled), 0)
            labels_vec = torch.cat((labels_vec, labels), 0)
            mu_mat = torch.cat((mu_mat, mu), 0)    
        # plot 10 first images + their reconstructions
    if 1:
        plt.figure(1)
        for i in range(10):           
            plt.subplot(2, 10, i+1)
            plt.title(labels[i].item())
            plt.imshow(tensor_images[i].detach().numpy().reshape(28, 28), cmap='gray')
            plt.subplot(2, 10, i+11)
            plt.title(labels[i].item())
            plt.imshow(reconstruction[i].detach().numpy().reshape(28, 28), cmap='gray')
        plt.show()
        
        # plot lataent space in 2 first dimensions (2D) with labels by color
        plt.figure(2)
        plt.scatter(z_sampled_mat[:, 0].detach().numpy(), z_sampled_mat[:, 1].detach().numpy(), c=labels_vec, cmap='tab10')
        plt.colorbar()
        plt.show() 
        
    return reconstructions, mu_mat

def choose_labeled_data_for_svm(labeled_data_len, labeled_train, train_labels, n_labels):


    # choose data with the chosen labels and their reconstructions
    # validae an equal number of data for each label
    chosen_z_samples = torch.empty(0, N_z)

    chosen_labels_vec = torch.empty(0)
    for i in range(n_labels):
        idx = np.where(train_labels == i)
        random.shuffle(idx)
        idx = idx[0][:labeled_data_len//n_labels]
        chosen_z_samples = torch.cat((chosen_z_samples, labeled_train[idx]), 0)
        chosen_labels_vec = torch.cat((chosen_labels_vec, train_labels[idx]), 0)
    return chosen_z_samples, chosen_labels_vec

def svm_after_vae(mu_train_labeled, labels_train, mu_test, labels_test, option):
    # svm algorithm after vae
    clf = []
    # best kernel for svm
    clf = svm.SVC(kernel='rbf', C=4)


    # fit the model with the labeled data
    clf.fit(mu_train_labeled, labels_train)

    if Save_model_to_file:
        # save the model to a file
        torch.save(clf, 'svm_model_{}.pth'.format(option))

    # predict the labels of the test set
    predicted_labels_test = torch.tensor(clf.predict(mu_test))

    # compute the accuracy
    svc_accuracy_test = metrics.accuracy_score(labels_test, predicted_labels_test)
    print('SVM test accuracy with {} labels: {:.4f} %'.format(option, svc_accuracy_test*100))

    return svc_accuracy_test



def main(label_options):
    results = {}


    print('Loading data...')
    train_loader, test_loader = load_data_fashion()  

    # load the trained model from a file
    if Load_model_from_file:
        # load the trained model from a file
        print('Loading trained model from file...')
        trained_model = LinearVAE()
        trained_model.load_state_dict(torch.load('trained_model.pth'))

    else:
        
        print('Generating model...')
        model = LinearVAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=Learning_rate)

        print('Training model...')
        loss, trained_model = train_model_vae(model, optimizer, train_loader)
        results['train_loss'] = loss

    if Save_model_to_file:
        # save the trained model to a file
        print('Saving trained model to file...')
        torch.save(trained_model.state_dict(), 'trained_model.pth')
    
    

    print('Testing model...')
    if Load_model_from_file:
        
        print('Loading results from file...')
        final_reconstructions_train = torch.load('final_reconstructions_train.pth')
        mu_mat_train = torch.load('mu_mat_train.pth')
        final_reconstructions_test = torch.load('final_reconstructions_test.pth')
        mu_mat_test = torch.load('mu_mat_test.pth')
        final_train_labels = torch.load('final_train_labels.pth')
        final_test_labels = torch.load('final_test_labels.pth')


    else:
        final_reconstructions_train, mu_mat_train = test_model(trained_model, train_loader)
        final_reconstructions_test, mu_mat_test = test_model(trained_model, test_loader)
        final_train_labels = train_loader.dataset.targets
        final_test_labels = test_loader.dataset.targets
    
    if Save_model_to_file:
        # save the results to a file
        print('Saving results to file...')
        torch.save(final_reconstructions_train, 'final_reconstructions_train.pth')
        torch.save(mu_mat_train, 'mu_mat_train.pth')
        torch.save(final_reconstructions_test, 'final_reconstructions_test.pth')
        torch.save(mu_mat_test, 'mu_mat_test.pth')
        torch.save(final_train_labels, 'final_train_labels.pth')
        torch.save(final_test_labels, 'final_test_labels.pth')



    for labeled_data_len in label_options:

        #print('Choosing data for SVM... N-labels = '+ str(labeled_data_len))
        # apply SVM on the chosen data using the mu vector from the latent space    
        chosen_mu_train, chosen_labels_train = choose_labeled_data_for_svm(labeled_data_len, mu_mat_train, final_train_labels, N_labels)

        #print('Appling SVM... N-labels = '+ str(labeled_data_len))
        # apply SVM on the chosen data using the mu vector from the latent space
        accuracy = svm_after_vae(chosen_mu_train, chosen_labels_train, mu_mat_test, final_test_labels, labeled_data_len)

        results['accuracy'] = results.get('accuracy', []) + [accuracy]
    
    return results

# Run the model
options = [3000,1000,600,100] # number of labeled data

if __name__ == "__main__":
    results = main(options) # run the model


