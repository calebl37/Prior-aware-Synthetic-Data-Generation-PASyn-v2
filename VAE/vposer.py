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
        -input_dim (int): number of neurons in input layer
        -output_dim (int): number of neurons in output layer
        -hidden_neurons (list): hidden_neurons[i] = number of neurons in hidden layer i
        -output_dim (int): number of neurons in output layer
    Returns:
        nn.Module: A PyTorch nn.Sequential object
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
    '''
    A custom loss function for a VAE that balances reconstruction loss 
    (SSE between the original input and VAE output) with KL divergence loss 
    (distance between the latent space and a normal distribution)
    
    Args:
        -w1 (float): weight of the KL divergence loss term
        -w2 (float): weight of the reconstruction loss term

    '''
    def __init__(self, w1: float=0.005, w2:float=0.01):
        super().__init__()
        self.w1=w1
        self.w2=w2

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        '''
        PyTorch forward pass
        
        Args:
            -x (tensor): shape (n, n_leg_joints * 3), the original leg pose XYZ angles 
            -x_hat (tensor): shape (n, n_leg_joints * 3), the reconstructed leg pose XYZ angles 
            -mu (tensor): shape (latent_dim, ), the mean of the latent space
            -logvar: shape (latent_dim, ), the variance of the latent space
        Returns:
            tensor: loss with gradient attached
        '''
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (self.w1 * recon_loss + self.w2 * kl_loss)

class VPoser(nn.Module):
    
    '''
    A PyTorch nn.Module implementation of a variational autoencoder with a custom loss function
    
    Args:
        -n_leg_joints (int): number of leg joints of the animal
        -hidden_neurons (list): number of neurons in each layer of the encoder, and reversed for the decoder
        -latent_dim (int): dimension of the latent space

    '''
    def __init__(self, n_leg_joints: int = 36, hidden_neurons: list = [32, 24], latent_dim: int = 20):
        super().__init__()

        #build encoder 
        self.encoder = fully_connected_block(input_dim = n_leg_joints, output_dim= hidden_neurons[-1], hidden_neurons=hidden_neurons[:-1])
        
        #generate the mean and variance from the latent space
        self.fc_mu = nn.Linear(hidden_neurons[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_neurons[-1], latent_dim)

        #build decoder
        self.decoder = fully_connected_block(input_dim = latent_dim, output_dim= n_leg_joints, hidden_neurons=hidden_neurons[::-1])

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        '''
        The reparameterization trick to sample from the latent space
        Args:
            -mu (tensor): shape (latent_dim, ), the mean of the latent space
            -logvar: shape (latent_dim, ), the variance of the latent space
        Returns:
            tensor: a random sample of shape (latent_dim, ) from a normal distribution shifted by mu and scaled by logvar
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        PyTorch forward pass
        
        Args:
            -x (tensor): shape (n, n_leg_joints * 3), the original leg pose XYZ angles 
        Returns:
            tensor: shape (n, n_leg_joints * 3), the reconstructed leg pose XYZ angles 
        '''

        #encode the input in the latent space
        encoder_out = self.encoder(x)

        #get the mean and variance from the latent space
        mu = self.fc_mu(encoder_out)
        logvar = self.fc_logvar(encoder_out)

        #apply the reparameterization trick to sample a vector from the latent space
        z = self.reparameterize(mu, logvar)

        #decode the sample
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

class VPoserWrapper:
    '''
    A scikit-learn wrapper for the VPoser PyTorch model that implements fit() and predict(). Handles training on existing leg poses and generation of new leg poses, 
    as well as hyperparameter tuning, seen and unseen loss plotting, and model checkpoint saving.
    
    Args:
        -device: (torch.device): cpu or gpu
        -n_leg_joints (int): number of leg joints of the animal
        -hidden_neurons (list): number of neurons in each layer of the encoder, and reversed for the decoder
        -latent_dim (int): dimension of the latent space
        -lr (float): learning rate for VPoser training
        -epochs (int): number of forward and backward passes in the training loop
        -batch_size (int): size per batch in training
        -w1 (float): weight of the KL divergence loss term
        -w2 (float): weight of the reconstruction loss term
        -checkpoint_path (str): path to save model checkpoints

    '''
    def __init__(self, device: torch.device, n_leg_joints: int = 36, hidden_neurons: list = [32, 24], latent_dim: int = 20, lr:float=1e-3, 
                 epochs: int=250, batch_size: int=128 , w1: float = 0.005, w2: float = 0.01, checkpoint_path: str="checkpoint.pt"):
        self.device = device
        self.n_leg_joints = n_leg_joints

        self.latent_dim = latent_dim

        self.model = VPoser(n_leg_joints=self.n_leg_joints, hidden_neurons=hidden_neurons, latent_dim=self.latent_dim).to(self.device)
        self.lr = lr 
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.w1 = w1 
        self.w2 = w2 
        self.checkpoint_path = checkpoint_path
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)


    def load_checkpoint(self):
        '''
        Instantiates this model using a saved checkpoint if there is one
        '''
        #load checkpoint if there is one
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location = self.device, weights_only=False)
                        
        
            
            self.current_epoch = checkpoint['epoch']


            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']

            #check for mismatching architecture with checkpoint
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except (RuntimeError, ValueError):
                print("Error: Existing checkpoint has a different architecture than this instantiation. Either remove the old checkpoint and re-run, or re-instantiate this class with the same architecture used in the previous save.")
                return 


    def fit(self, X_train: torch.Tensor, X_val: torch.Tensor) -> None:

        '''
        Handles training of VPoser, using the specified hyperparameters, saves the model state periodically,
        records train and validation loss over time

        Args:
            X_train (tensor): the XYZ angles of seen leg poses to train on 
            X_val (tensor): the XYZ angles of unseen leg poses to evaluate on 

        Returns:
            None
        '''
        #GPU support
        X_train = X_train.to(self.device)
        X_val = X_val.to(self.device)

        #Custom loss function 
        criterion = VAELoss(w1=self.w1, w2=self.w2)

        train_tensor = TensorDataset(X_train)
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)

        #track per-epoch seen and unseen loss
        self.train_losses = []
        self.val_losses = []

        self.current_epoch = 0
        
        #load checkpoint if there is one
        self.load_checkpoint()

        #training loop
        for i in range(self.current_epoch, self.epochs):

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
                self.optimizer.step()
                self.optimizer.zero_grad()

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
                    'epoch': i+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                },
                self.checkpoint_path
            )

            if (i + 1) % (self.epochs // 10) == 0:
                print("Checkpoint saved! Epoch {} Complete: VAE Custom Train Loss = {}, Validation Loss = {}".format(i+1, avg_train_loss, val_loss))
            
    def predict(self, X) -> torch.Tensor:

        '''
        Generates new lew poses from the given ones, using the last saved checkpoint of the VPoser model

        Args:
            X (tensor): shape (m, n_leg_joint*3), the XYZ angles of seen leg poses to generate new ones from
           
        Returns:
            tensor: shape (m, n_leg_joint*3), the XYZ angles of new leg poses generated by VPoser
        '''
        #GPU support
        X = X.to(self.device)
        #load checkpoint if there is one
        self.load_checkpoint()

        #get the direct VAE reconstruction with a single forward pass
        self.model.eval()
        with torch.no_grad():
            output, _, _ = self.model(X)
            return output
        
    def plot_losses(self) -> None:

        '''
        Uses the last saved checkpoint of VPoser to plot the train and test loss over each epoch so far. 
        Saves plot in the same directory

        Args:
            None
        Returns:
            None
        '''

        #load checkpoint if there is one
        self.load_checkpoint()

        epochs = np.arange(self.current_epoch)
        plt.title("Train/Val VAE Loss over time for w1 = {} and w2 = {}".format(self.w1, self.w2))
        plt.plot(epochs, self.train_losses, c='r', label="Train")
        plt.plot(epochs, [v.cpu() for v in self.val_losses], c='b', label="Validation")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("VAE Custom Loss")
        plt.savefig("losses.jpg")
        plt.show()
        
