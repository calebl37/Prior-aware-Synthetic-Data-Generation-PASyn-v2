import os
import numpy as np
import torch
import torchvision.datasets as dset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from cnn_adain_model import ConvStyleTransfer
import math
import random


if __name__ == "__main__":

    # Get style transfer hyperparameters from the command line
    parser = argparse.ArgumentParser(description='Train AdaIN style transfer model')
    parser.add_argument('--height', type=int, default = 128, help='Image height (default: 128)')
    parser.add_argument('--width', type=int, default = 128, help='Image width (default: 128)')
    parser.add_argument('--epochs', type=int, default = 5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--alpha', type=float, default=0.1, help='Style interpolation weight (0-1)')
    parser.add_argument('--l_content_weight', type=float, default = 1.0, help='Weight for content loss')
    parser.add_argument('--l_style_weight', type=float, default=10.4, help='Weight for style loss')
    parser.add_argument('--content_dir', type=str, help='Directory for content images (optional, for paired training)')
    parser.add_argument('--style_dir', type=str, help='Directory for style images (optional, for paired training)')
    parser.add_argument('--train_dir', type=str, default='celebA', help='Directory for training dataset (for self-training, default: celebA)')
    parser.add_argument('--seed', type=int, default=999, help='Random seed for reproducibility (default: 999)')
    args = parser.parse_args()._get_kwargs()
    kwargs = {i: j for i, j in args}
    
    # Set random seed for reproducibility
    seed = kwargs.pop('seed')
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    print(f"Random Seed: {seed}")


    # Load training data
    # Option 1: Paired training (content_dir and style_dir specified)
    # Option 2: Self-training on single dataset (split into content/style)

    content_dir = kwargs.pop('content_dir')
    style_dir = kwargs.pop('style_dir')
    if content_dir and style_dir:
        # Load paired content and style images
        print("Loading paired datasets...")
        print(f"Loading content images from: {content_dir}")
        content_dataset = dset.ImageFolder(
            root=content_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((kwargs['height'], kwargs['width']))
            ])
        )
        content_images = torch.stack([t[0] for t in content_dataset], dim=0)
        if content_images.shape[0] < 2:
            print("Not enough images")
            exit()
        
        print(f"Loading style images from: {style_dir}")
        style_dataset = dset.ImageFolder(
            root=style_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((kwargs['height'], kwargs['width']))
            ])
        )
        style_images = torch.stack([t[0] for t in style_dataset], dim=0)
        if style_images.shape[0] < 2:
            print("Not enough images")
            exit()
        
        # Match dataset sizes
        if content_images.shape[0] > style_images.shape[0]:
            n_repeats = math.ceil(content_images.shape[0] / style_images.shape[0])
            style_images = style_images.repeat(n_repeats, 1, 1, 1)[:content_images.shape[0]]
        elif style_images.shape[0] > content_images.shape[0]:
            n_repeats = math.ceil(style_images.shape[0] / content_images.shape[0])
            content_images = content_images.repeat(n_repeats, 1, 1, 1)[:style_images.shape[0]]
            
    else:
        # Load single dataset and split into content/style
        train_dir = kwargs.pop('train_dir')
        print(f"Loading training dataset from: {train_dir}")
        train_dataset = dset.ImageFolder(
            root=train_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((kwargs['height'], kwargs['width']))
            ])
        )
        
        all_images = torch.stack([t[0] for t in train_dataset], dim=0)
        print(f"Loaded {all_images.shape[0]} images")
        
        # Split dataset into content and style (as done in notebook)
        if all_images.shape[0] < 2:
            print("Not enough images")
            exit()

        #ensure there are an even number of training images
        if all_images.shape[0] % 2 == 1:
            all_images = all_images[1:]

        split_idx = int(all_images.shape[0] * 0.5)
        content_images = all_images[:split_idx]
        style_images = all_images[split_idx:]
        print(f"Split into {content_images.shape[0]} content images and {style_images.shape[0]} style images")

    print(f"\nContent images shape: {content_images.shape}")
    print(f"Style images shape: {style_images.shape}")

    # GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice = {device}")

    # Initialize the style transfer model with parsed hyperparameters
    style_transfer = ConvStyleTransfer(device=device, **kwargs)

    # Train the style transfer model
    print("\nStarting training...")
    print(f"Training parameters:")
    print(f"  - Input Image size: {style_transfer.height}x{style_transfer.width}")
    print(f"  - Epochs: {style_transfer.epochs}")
    print(f"  - Batch size: {style_transfer.batch_size}")
    print(f"  - Learning rate: {style_transfer.lr}")
    print(f"  - Alpha (style interpolation): {style_transfer.alpha}")
    print(f"  - Content weight: {style_transfer.l_content_weight}")
    print(f"  - Style weight: {style_transfer.l_style_weight}")
    
    style_transfer.fit(X=content_images, y=style_images)

    # Plot and save training losses
    print("\nTraining completed!")
    style_transfer.plot_losses()

