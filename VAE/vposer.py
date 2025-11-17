import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader


def fully_connected_block(input_dim:int=32, output_dim: int=2, hidden_neurons:list=[16, 8, 4]) -> nn.Module:
    '''
    A basic PyTorch Feed Forward Neural Network with LeakyReLU activation functions
    
    Args:
        -input_dim: number of neurons in input layer
        -output_dim: number of neurons in output layer
        -hidden_neurons:
    
    
    '''
    layers = []
    layers.append(nn.Linear(input_dim, hidden_neurons[0]))
    layers.append(nn.LeakyReLU())
    for i in range(0, len(hidden_neurons)-1):
        layers.append(nn.Linear(hidden_neurons[i], hidden_neurons[i+1]))
        layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(hidden_neurons[-1], output_dim))
    return nn.Sequential(*layers)


class VAELoss(nn.Module):
    def __init__(self, w1: float=0.005, w2:float=0.01):
        super().__init__()
        self.w1=w1
        self.w2=w2

    def forward(self, x, x_hat, mu, logvar):
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (self.w1 * recon_loss + self.w2 * kl_loss)

class VPoser(nn.Module):
    def __init__(self, n_leg_joints: int = 36, hidden_neurons: list = [32, 24], latent_dim: int = 20):
        super().__init__()

        self.encoder = fully_connected_block(input_dim = n_leg_joints, output_dim= hidden_neurons[-1], hidden_neurons=hidden_neurons[:-1])
        self.fc_mu = nn.Linear(hidden_neurons[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_neurons[-1], latent_dim)
        self.decoder = fully_connected_block(input_dim = latent_dim, output_dim= n_leg_joints, hidden_neurons=hidden_neurons[::-1])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoder_out = self.encoder(x)
        mu = self.fc_mu(encoder_out)
        logvar = self.fc_logvar(encoder_out)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

class VPoserWrapper:
    def __init__(self, device: torch.device, n_leg_joints: int = 36, hidden_neurons: list = [32, 24], latent_dim: int = 20, lr:float=1e-3, 
                 epochs: int=10, batch_size: int=64, w1: float = 0.005, w2: float = 0.01):
        self.device = device
        self.n_leg_joints = n_leg_joints
        self.latent_dim = latent_dim
        self.model = VPoser(n_leg_joints=n_leg_joints, hidden_neurons=hidden_neurons, latent_dim=self.latent_dim)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.w1 = w1
        self.w2 = w2



    def fit(self, X_train: torch.Tensor, X_val: torch.Tensor):

        #GPU support
        X_train = X_train.to(self.device)
        X_val = X_val.to(self.device)

        #Custom loss function 
        criterion = VAELoss(w1=self.w1, w2=self.w2)
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        train_tensor = TensorDataset(X_train)
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)

        #track per-epoch seen and unseen loss
        self.train_losses = []
        self.val_losses = []

        current_epoch = 0
        
        #load checkpoint if there is one
        if os.path.exists("checkpoint.pt"):
            checkpoint = torch.load("checkpoint.pt", map_location = self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            current_epoch = checkpoint['epoch']


            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']

        #training loop
        for i in range(current_epoch+1, self.epochs):

            self.model.train()
            total_loss = 0
            for batch in train_loader:   
                imgs = batch[0]

                #output reconstruction plus latent space parameters
                x_hat, mu, logvar = self.model(imgs)

                #compute batch VAE loss
                loss: torch.Tensor = criterion(imgs, x_hat, mu, logvar)

                #backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
            
            #record train loss
            avg_train_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            #record validation loss
            self.model.eval()
            with torch.no_grad():
                x_hat, mu, logvar = self.model(X_val)
                val_loss = criterion(X_val, x_hat, mu, logvar)
                self.val_losses.append(val_loss)

            torch.save(
                {
                    'epoch': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                },
                "checkpoint.pt"
            )

            if (i + 1) % (self.epochs // 10) == 0:
                print("Checkpoint saved! Epoch {} Complete: VAE Custom Train Loss = {}, Validation Loss = {}".format(i+1, avg_train_loss, val_loss))
            
    def predict(self, X) -> torch.Tensor:
        #load checkpoint if there is one
        if os.path.exists("checkpoint.pt"):
            checkpoint = torch.load("checkpoint.pt", map_location = self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        with torch.no_grad():
            output, _, _ = self.model(X)
            return output
        
    def plot_losses(self):

        #load checkpoint if there is one
        if os.path.exists("checkpoint.pt"):
            checkpoint = torch.load("checkpoint.pt", map_location = self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            current_epoch = checkpoint['epoch']

            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']

        epochs = np.arange(current_epoch)
        plt.title("Train/Val VAE Loss over time for w1 = {} and w2 = {}".format(self.w1, self.w2))
        plt.plot(epochs, self.train_losses, c='r', label="Train")
        plt.plot(epochs, self.val_losses, c='b', label="Validation")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("VAE Custom Loss")
        plt.savefig("losses.jpg")
        plt.show()
        
