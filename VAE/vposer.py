import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def fully_connected_block(input_dim:int=32, output_dim: int=2, hidden_neurons:list=[16, 8, 4]):
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
    def __init__(self, n_leg_joints: int = 36, hidden_neurons: list = [32, 24], latent_dim: int = 20, lr:float=1e-3, 
                 epochs: int=10, batch_size: int=64, optimizer_func: torch.optim.Optimizer = torch.optim.Adam):
        self.n_leg_joints = n_leg_joints
        self.latent_dim = latent_dim
        self.model = VPoser(n_leg_joints=n_leg_joints, hidden_neurons=hidden_neurons, latent_dim=self.latent_dim)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_func = optimizer_func


    def fit(self, X):

        criterion = VAELoss()
        optimizer = self.optimizer_func(self.model.parameters(), self.lr)

        train_tensor = TensorDataset(X)
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)

        for i in range(self.epochs):

            total_loss = 0
            self.model.train()
            for batch in train_loader:   
                imgs = batch[0]
                x_hat, mu, logvar = self.model(imgs)
                loss = criterion(imgs, x_hat, mu, logvar)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print("Epoch {} Complete: VAE Custom Loss = {}".format(i+1, total_loss / len(train_loader)))
            
    def generate_poses(self, X):
        self.model.eval()
        with torch.no_grad():
            output, _, _ = self.model(X)
            return output