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


def rgba_loader(path):
    """Custom loader for RGBA images."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGBA")


if __name__ == "__main__":

    # Get style transfer hyperparameters from the command line
    parser = argparse.ArgumentParser(description='Train AdaIN style transfer model')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--alpha', type=float, help='Style interpolation weight (0-1)')
    parser.add_argument('--content_weight', type=float, help='Weight for content loss')
    parser.add_argument('--style_weight', type=float, help='Weight for style loss')
    parser.add_argument('--hidden_channels', type=int, nargs='+', help='Hidden channel sizes for encoder/decoder')
    parser.add_argument('--height', type=int, help='Image height (default: 64)')
    parser.add_argument('--width', type=int, help='Image width (default: 64)')
    parser.add_argument('--output_channels', type=int, help='Output channels for encoder (default: 512)')
    parser.add_argument('--content_dir', type=str, help='Directory for content images (optional, for paired training)')
    parser.add_argument('--style_dir', type=str, help='Directory for style images (optional, for paired training)')
    parser.add_argument('--train_dir', type=str, default='celebA', help='Directory for training dataset (for self-training, default: celebA)')
    parser.add_argument('--split_ratio', type=float, default=0.5, help='Ratio for splitting training dataset into content/style (default: 0.5)')
    parser.add_argument('--seed', type=int, default=999, help='Random seed for reproducibility (default: 999)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    print(f"Random Seed: {args.seed}")

    # Parse arguments into kwargs for model initialization
    kwargs = {}
    if args.epochs is not None:
        kwargs['epochs'] = args.epochs
    if args.lr is not None:
        kwargs['lr'] = args.lr
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    if args.alpha is not None:
        kwargs['alpha'] = args.alpha
    if args.content_weight is not None:
        kwargs['content_weight'] = args.content_weight
    if args.style_weight is not None:
        kwargs['style_weight'] = args.style_weight
    if args.hidden_channels is not None:
        kwargs['hidden_channels'] = args.hidden_channels
    if args.output_channels is not None:
        kwargs['output_channels'] = args.output_channels
    
    # Set image dimensions (default: 64x64)
    height = args.height if args.height is not None else 64
    width = args.width if args.width is not None else 64
    kwargs['height'] = height
    kwargs['width'] = width

    # Load training data
    # Option 1: Paired training (content_dir and style_dir specified)
    # Option 2: Self-training on single dataset (split into content/style)
    if args.content_dir and args.style_dir:
        # Load paired content and style images
        print("Loading paired datasets...")
        print(f"Loading content images from: {args.content_dir}")
        content_dataset = dset.ImageFolder(
            root=args.content_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((height, width))
            ])
        )
        content_images = torch.stack([t[0] for t in content_dataset], dim=0)
        
        print(f"Loading style images from: {args.style_dir}")
        # Try RGBA loader first (for PNG images with alpha channel)
        try:
            style_dataset = dset.ImageFolder(
                root=args.style_dir,
                loader=rgba_loader,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.CenterCrop((height, width))
                ])
            )
            style_images = torch.stack([t[0][:3, :, :] for t in style_dataset], dim=0)
        except:
            # Fall back to regular RGB loader
            style_dataset = dset.ImageFolder(
                root=args.style_dir,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((height, width))
                ])
            )
            style_images = torch.stack([t[0] for t in style_dataset], dim=0)
        
        # Match dataset sizes
        if content_images.shape[0] > style_images.shape[0]:
            n_repeats = math.ceil(content_images.shape[0] / style_images.shape[0])
            style_images = style_images.repeat(n_repeats, 1, 1, 1)[:content_images.shape[0]]
        elif style_images.shape[0] > content_images.shape[0]:
            n_repeats = math.ceil(style_images.shape[0] / content_images.shape[0])
            content_images = content_images.repeat(n_repeats, 1, 1, 1)[:style_images.shape[0]]
            
    else:
        # Load single dataset and split into content/style
        print(f"Loading training dataset from: {args.train_dir}")
        train_dataset = dset.ImageFolder(
            root=args.train_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((height, width))
            ])
        )
        
        all_images = torch.stack([t[0] for t in train_dataset], dim=0)
        print(f"Loaded {all_images.shape[0]} images")
        
        # Split dataset into content and style (as done in notebook)
        split_idx = int(all_images.shape[0] * args.split_ratio)
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
    print(f"  - Epochs: {style_transfer.epochs}")
    print(f"  - Batch size: {style_transfer.batch_size}")
    print(f"  - Learning rate: {style_transfer.lr}")
    print(f"  - Alpha (style interpolation): {style_transfer.alpha}")
    print(f"  - Content weight: {style_transfer.content_weight}")
    print(f"  - Style weight: {style_transfer.style_weight}")
    print(f"  - Image size: {height}x{width}")
    
    style_transfer.fit(X=content_images, y=style_images)

    # Plot and save training losses
    print("\nTraining completed!")
    if hasattr(style_transfer, 'losses') and len(style_transfer.losses) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(style_transfer.losses, 'b-', label='Training Loss')
        plt.title(f"Style Transfer Training Loss\n(Content Weight={style_transfer.content_weight}, Style Weight={style_transfer.style_weight})")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("style_transfer_losses.jpg")
        print(f"Loss plot saved: style_transfer_losses.jpg")
        print(f"Final average loss: {np.mean(style_transfer.losses[-100:]):.4f}")
    else:
        print("No loss history available to plot.")

