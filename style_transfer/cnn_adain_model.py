import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader

def conv_encoder(input_channels: int, output_channels: int, hidden_channels: list[int], add_bias = False) -> nn.Module:
    layers = []
    layers.append(nn.Conv2d(input_channels, hidden_channels[0], kernel_size=4, stride=2, padding=1, bias=add_bias))
    layers.append(nn.LeakyReLU(0.2, True))
    for k in range(0, len(hidden_channels)-1):
        layers.append(nn.Conv2d(hidden_channels[k], hidden_channels[k+1], kernel_size=4, stride=2, padding=1, bias=add_bias))
        layers.append(nn.BatchNorm2d(hidden_channels[k+1]))
        layers.append(nn.LeakyReLU(0.2, True))
    layers.append(nn.Conv2d(hidden_channels[-1], output_channels, kernel_size=4, stride=2, padding=1, bias=add_bias))
    return nn.Sequential(
            *layers
    )

def conv_decoder(input_channels: int, output_channels: int, hidden_channels: list[int], add_bias: bool = False) -> nn.Module:

    layers = []
    layers.append(nn.ConvTranspose2d(input_channels, hidden_channels[0], kernel_size=4, stride=2, padding=1, bias=add_bias))
    layers.append(nn.BatchNorm2d(hidden_channels[0]))
    layers.append(nn.ReLU(True))
    for k in range(0, len(hidden_channels)-1):
        layers.append(nn.ConvTranspose2d(hidden_channels[k], hidden_channels[k+1], kernel_size=4, stride=2, padding=1, bias=add_bias))
        layers.append(nn.BatchNorm2d(hidden_channels[k+1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.ConvTranspose2d(hidden_channels[-1], output_channels, kernel_size=4, stride=2, padding=1, bias=add_bias))

    return nn.Sequential(
            *layers
    )

def calc_content_loss(out_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    return torch.mean((out_feat - target_feat) ** 2)
 
def calc_style_loss(out_feats: torch.Tensor, style_feats: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for of, sf in zip(out_feats, style_feats):
        o_mean, o_std = of.mean([2, 3]), of.std([2, 3])
        s_mean, s_std = sf.mean([2, 3]), sf.std([2, 3])
        loss += torch.mean((o_mean - s_mean)**2) + torch.mean((o_std - s_std)**2)
    return loss

# AdaIN function
def adain(content_feat: torch.Tensor, style_feat: torch.Tensor, eps=1e-5) -> torch.Tensor:
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True)
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True)
    return s_std * (content_feat - c_mean) / (c_std + eps) + s_mean


class ConvStyleTransfer:
    
    def __init__(self, device: torch.device, input_channels: int=3, output_channels:int=512, hidden_channels: list = [128, 256], 
                 height:int=64, width:int=64, alpha: float=0.5, content_weight: float = 1.0, style_weight: float = 10.4,
                 epochs:int=10, lr:float=1e-3, batch_size: int=64):
        self.device = device
        self.encoder = conv_encoder(input_channels, output_channels, hidden_channels)
        self.decoder = conv_decoder(output_channels, input_channels, hidden_channels[::-1])
        self.model = nn.Sequential(self.encoder, self.decoder).to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.height = height
        self.width = width
        self.alpha = alpha
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.batch_size = batch_size
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):

        #GPU support
        X = X.to(self.device)
        y = y.to(self.device)
        
        
        dataloader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        
        current_epoch = 0
        
        #load checkpoint if there is one
        if os.path.exists("checkpoint.pt"):
            checkpoint = torch.load("checkpoint.pt", map_location = self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            current_epoch = checkpoint['epoch']

            self.losses = checkpoint['losses']
        
        
        #training loop
        for epoch in range(current_epoch+1, self.epochs):
            
            self.model.train()
            
            self.losses = []
            for content, style in dataloader:
    
                # Encode
                c_feat: torch.Tensor  = self.encoder(content)
                s_feat: torch.Tensor  = self.encoder(style)

                # AdaIN
                t_feat = c_feat * (1-self.alpha) + self.alpha * adain(c_feat, s_feat)

                # Decode
                output = self.decoder(t_feat)

                # Compute features again
                out_feat = self.encoder(output)

                #output encodings at each layer
                cum_out_feats = [self.encoder[:i+1](output) for i in range(len(self.encoder)-1)]
                
                #style encodings at each layer
                cum_style_feats = [self.encoder[:i+1](style) for i in range(len(self.encoder)-1)]

                # Loss: weighted sum of content loss and cumulative style loss per layer
                content_loss = calc_content_loss(out_feat, t_feat.detach())
                style_loss = calc_style_loss(cum_out_feats, cum_style_feats)
                loss = self.content_weight * content_loss + self.style_weight * style_loss
                
                self.losses.append(loss.item())

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': self.losses
                },
                "checkpoint.pt"
            )
            print("Checkpoint saved: Epoch {} complete: Average Loss = {}".format(epoch, np.mean(self.losses)))
            
            
    #styles the background (content) to match the zebra (style), and then composites the zebra with the stylized background
    #using the alpha channel (PNG foreground/background pixel map) of the zebra
    def stylize(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 0.1):
        
        #GPU support
        content = content.to(self.device)
        style = style.to(self.device)
        
        #load checkpoint if there is one
        if os.path.exists("checkpoint.pt"):
            checkpoint = torch.load("checkpoint.pt", map_location = self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
    
        with torch.no_grad():
            # Encode
            c_feat = self.encoder(content)
            s_feat = self.encoder(style)

            # AdaIN
            t_feat = (1-alpha) * c_feat + alpha * adain(c_feat, s_feat)

            # Decode
            stylized_images = self.decoder(t_feat)

            return stylized_images