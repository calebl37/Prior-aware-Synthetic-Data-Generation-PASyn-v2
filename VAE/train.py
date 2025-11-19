import os
import numpy as np
import torch
from vposer import VPoserWrapper
import argparse
import random

if __name__ == "__main__":

    #get VAE hyperparameters from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_leg_joints', type = int, default=36, help='Total number of leg joints')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--w1', type=float, default=0.005, help='Weight of KL divergence loss (default: 0.005)')
    parser.add_argument('--w2', type=float, default=0.01, help='Weight of reconstruction loss (default: 0.01)')
    parser.add_argument('--seed', type=int, default=5, help='Random seed for reproducibility (default: 5)')

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


    #load 3 datasets - the XYZ angles of leg poses from other low poly animated 3D animal models
    #skip the first 3 columns of every row because they are the XYZ angles of the T-pose (trivial)
    train_leg_poses_other_animals = torch.tensor(
    np.load(os.path.join("data", "train_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    test_leg_poses_other_animals = torch.tensor(
    np.load(os.path.join("data", "test_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    val_leg_poses_other_animals = torch.tensor(
    np.load(os.path.join("data", "val_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    #GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device =", device)

    #initialize the VAE with the parsed hyperparameters
    vposer = VPoserWrapper(device=device, **kwargs)

    #train the VAE
    vposer.fit(X_train=train_leg_poses_other_animals, X_val=val_leg_poses_other_animals)
    
    #save train and validation loss over time
    vposer.plot_losses()