import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
vgg16_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)


def conv_encoder(input_channels: int, output_channels: int, hidden_channels: list[int], add_bias = False) -> nn.Module:
    '''
    A sequence of 2D convolutional layers that each halve the height and width
    Interleaves with 2D Batch Normalization layers and Leaky ReLU activation functions
    
    Args:
        input_channels (int): number of channels in the input image
        output_channels (int): number of channels in the output image
        hidden_channels (list): number of channels in each intermediary convolutional layer
                                output_size = input_size // (2 ** (n_hidden_channels + 1))
        add_bias (bool): whether or not to add a bias term to each convolutional layer
    Returns:
        nn.Sequential Module
    '''
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
    '''
    A sequence of 2D convolutional transpose layers that each double the height and width
    Interleaves with 2D Batch Normalization layers and ReLU activation functions
    
    Args:
        input_channels (int): number of channels in the input image
        output_channels (int): number of channels in the output image
        hidden_channels (list): number of channels in each intermediary convolutional layer
                                output_size = input_size * (2 ** (n_hidden_channels + 1))
        add_bias (bool): whether or not to add a bias term to each convolutional layer
    Returns:
        nn.Sequential Module
    '''
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

    '''
    SSE between the encoded feature maps of the stylized output and the encoded feature maps of the original content image,
    to enforce overall consistency between the original and stylized images

    Args:
        out_feat (tensor): shape (N, C, H, W) the encoded feature maps of the stylized output
        target_feat (tensor): shape (N, C, H, W) the encoded feature maps of the original content image

    Returns:
        scalar SSE between out_feat and target_feat
    '''
    return torch.mean((out_feat - target_feat) ** 2)
 
def calc_style_loss(out_feats: torch.Tensor, style_feats: torch.Tensor) -> torch.Tensor:

    '''
    MSE between per-pixel encoding means of the original and stylized images and MSE between per-pixel encoding standard deviations 
    of the original and stylized images, summed across all encoding layers,
    to enforce style consistency between the original and stylized image encodings at every layer

    Args:
        out_feats (list): the encoding of the stylized output at each layer i, each of shape (N, C_i, H_i, W_i) 
        style_feats (list): the encoding of the original content image at each layer i, each of shape (N, C_i, H_i, W_i) 

    Returns:
        scalar SSE between out_feat and target_feat
    '''


    loss = 0.0

    #for each layer of the encoder
    for of, sf in zip(out_feats, style_feats):

        #compute the per-pixel mean and standard deviation of the stylized output encoding at that layer 
        #shape (N, c_i, 1, 1)
        o_mean, o_std = of.mean([2, 3]), of.std([2, 3])

        #compute the per-pixel mean and standard deviation of the original content image encoding at that layer
        #shape (N, c_i, 1, 1) 
        s_mean, s_std = sf.mean([2, 3]), sf.std([2, 3])

        #accumulate the MSE between the original and stylized output encoding per-pixel means 
        #and the MSE between the original and stylized output encoding per-pixel standard deviations
        loss += torch.mean((o_mean - s_mean)**2) + torch.mean((o_std - s_std)**2)
    return loss

# AdaIN function
def adain(content_feat: torch.Tensor, style_feat: torch.Tensor, eps=1e-5) -> torch.Tensor:
    '''
    PyTorch implementation of Adaptive Instance Normalization:
    Normalize the per-channel content feature maps to mean of 0 and unit variance, 
    shift by the style per_channel mean, scale by the style per_channel variation

    Args: 
        - content_feat (tensor): shape (n, c, h, w), the feature maps of the encoded content image
        - style_feat (tensor): shape (n, c, h, w), the feature maps of the encoded style image
    Returns:
        - tensor of shape (n, c, h, w), the content feature map normalized by the style feature map
    '''

    #content per-channel mean (n, c, 1, 1)
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)

    #content per-channel standard deviation (n, c, 1, 1)
    c_std = content_feat.std(dim=[2, 3], keepdim=True)

    #style per-channel mean (n, c, 1, 1)
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)

    #style per-channel standard deviation (n, c, 1, 1)
    s_std = style_feat.std(dim=[2, 3], keepdim=True)

    #subtract the content mean, divide by content standard deviation, scale by the style standard deviation. shift by the style mean
    # (n, c, 1, 1) * ((n, c, h, w) - (n, c, 1, 1)) / (n, c, 1, 1) + (n, c, 1, 1) = (n, c, h, w)
    return s_std * (content_feat - c_mean) / (c_std + eps) + s_mean


def reshape_image_block(input_h: int, input_w: int, input_channels: int, 
                        output_h: int, output_w: int, output_channels: int) -> nn.Module:
    
    '''
    A simple PyTorch module for resizing a batch of images from (N, C_in, H_in, W_in) to (N, C_out, H_out, W_out)
    without any convolutions

    Args:
        input_h (int): the given image height 
        input_w (int): the given image width
        input_channels (int): the number of given channels
        output_h (int): the desired image height 
        output_w (int): the desired image width
        output_channels (int): the number of desired channels

    Returns:
        nn.Sequential Module
    '''
    return nn.Sequential(nn.Flatten(start_dim=1), 
                         nn.Linear(input_channels*input_h*input_w, output_channels* output_h*output_w),
                         nn.Unflatten(dim=1, unflattened_size=(output_channels, output_h, output_w)))


class ConvStyleTransfer:
    

    '''
    A scikit-learn style wrapper for a CNN-encoder-decoder+AdaIN based image style transfer model that implements fit and predict. Handles training on given content and style images and blending given content and style images, 
    as well as hyperparameter tuning, seen and unseen loss plotting, and model checkpoint saving.

    Args:
        -device: (torch.device): cpu or gpu
        -height (int): input image height to the CNN-encoder-decoder+AdaIN
        -width (int): input image width to the CNN-encoder-decoder+AdaIN
        -input_channels (int): number of channels in the input image (3 for RGB)
        -output_channels (int): number of channels in the encoder output
        -hidden_channels (list): number of channels in each intermediary convolutional layer of the encoder (reversed for the decoder)
        -alpha (float): strength of AdAIN applied to the encoder output before being passed to decoder
        -l_content_weight (float): the weight of the content loss term in the custom loss function
        -l_style_weight (float): the weight of the style loss term in the custom loss function
        -epochs (int): number of forward and backward passes in the training loop
        -lr (float): learning rate for  CNN-encoder-decoder+AdaIN model training
        -batch_size (int): size per batch in training
        -checkpoint_path (str): path to save model checkpoints
    '''

    def __init__(self, device: torch.device, height:int=64, width:int=64, input_channels: int=3, output_channels:int=32, hidden_channels: list = [8, 16], alpha: float=0.5, l_content_weight: float = 1.0, l_style_weight: float = 10.4,
                 epochs:int=10, lr:float=1e-3, batch_size: int=64, checkpoint_path: str = "checkpoint.pt"):
        self.device = device
        self.height = height
        self.width = width
            
        #self.encoder = conv_encoder(input_channels, output_channels, hidden_channels)

        self.encoder = vgg16_model.features[:10]

        for p in self.encoder.parameters():
            p.requires_grad = False

        #self.decoder = conv_decoder(output_channels, input_channels, hidden_channels)

        self.decoder = conv_decoder(128, input_channels, [16])

        FIXED_SIZE = 64

        #we need the encoder to input size 64x64 and the decoder to output size 64x64, so reshape if needed
        if self.height != FIXED_SIZE or self.width != FIXED_SIZE:
            self.reshape_input_block = reshape_image_block(self.height, self.width, 3, FIXED_SIZE, FIXED_SIZE, 3)
            self.reshape_output_block = reshape_image_block(FIXED_SIZE, FIXED_SIZE, 3, self.height, self.width, 3)
            self.model = nn.Sequential(self.reshape_input_block, self.encoder, 
                                       self.decoder, self.reshape_output_block).to(self.device)
        else:
            self.model = nn.Sequential(self.encoder, self.decoder).to(self.device)

        

        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.l_content_weight = l_content_weight
        self.l_style_weight = l_style_weight
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    
    def load_checkpoint(self):

        '''
        Instantiates this model using a saved checkpoint if there is one
        '''

        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location = self.device, weights_only=False)
            
            self.current_epoch = checkpoint['epoch']

            self.losses = checkpoint['losses']

            self.height = checkpoint['height']

            self.width = checkpoint['width']

            #check for mismatching architecture with checkpoint
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except (RuntimeError, ValueError):
                print("Error: Cannot train due to existing checkpoint having a different architecture than this instantiation. Either remove the old checkpoint and train from scratch, or re-instantiate this class with the same architecture used in the previous save.")
                return 
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):

        '''
        Handles training of  CNN-encoder-decoder+AdaIN model, using the specified hyperparameters, saves the model state peridiodically,
        records content + style loss over time

        Args:
            X (tensor): the content images of shape (N, C, H, W)
            y (tensor): the style images of shape (N, C, H, W)

        Returns:
            None
        '''

        #GPU support
        X = X.to(self.device)
        y = y.to(self.device)
        
        
        dataloader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)
                
        self.current_epoch = 0

        self.losses = []
        
        #load checkpoint if there is one
        self.load_checkpoint()
            
        #training loop
        for epoch in range(self.current_epoch, self.epochs):
            
            self.model.train()
            
            epoch_losses = []
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
                loss = self.l_content_weight * content_loss + self.l_style_weight * style_loss
                
                epoch_losses.append(loss.item())

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()

            self.losses.append(np.mean(epoch_losses))
            
            #save checkpoint
            torch.save(
                {
                    'height': self.height,
                    'width': self.width,
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'losses': self.losses
                },
                self.checkpoint_path
            )
            print("Checkpoint saved: Epoch {} complete: Average Loss = {}".format(epoch, np.mean(self.losses)))
            

    def predict(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:

        '''
        Handles blending content and style

        Args:
            content (tensor): the content images to style, shape (N, C, H, W)
            style (tensor): the style images to blend with the content, shape (N, C, H, W)
            alpha (float): strength of AdAIN applied to the encoder output before being passed to decoder
        
        Returns:
            stylized images, tensor of shape (N, C. H, W)
            
        '''
        
        #GPU support
        content = content.to(self.device)
        style = style.to(self.device)

        #check for dimensionality mismatches
        assert content.shape[0] == style.shape[0] and content.shape[2] == self.height and content.shape[3] == self.width and style.shape[2] == self.height and style.shape[3] == self.width, "dimension mismatch with style and content images"
        
        #load checkpoint if there is one
        self.load_checkpoint()
        
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
        
    def plot_losses(self) -> None:

        '''
        Uses the last saved checkpoint of the CNN-encoder-decoder+AdaIN model to plot the train and test loss over each epoch so far. 
        Saves plot in the same directory

        Args:
            None
        Returns:
            None
        '''

        #load checkpoint if there is one
        self.load_checkpoint()

        epochs = np.arange(self.current_epoch)
        plt.title("Weighted Content Loss and Style Loss over time for l_content_weight = {} and l_style_weight = {}".format(self.l_content_weight, self.l_style_weight))
        plt.plot(epochs, self.losses, c='r')
        plt.xlabel("Epochs")
        plt.ylabel("Weighted Content Loss and Style Loss")
        plt.savefig("losses.jpg")
        plt.show()