import os
import numpy as np
import torch
from vposer import VPoserWrapper
import argparse

if __name__ == "__main__":

    #get VAE hyperparameters from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_leg_joints', type = int, default=36)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--w1', type=float, default=0.005)
    parser.add_argument('--w2', type=float, default=0.01)
    args = parser.parse_args()._get_kwargs()
    kwargs = {i: j for i, j in args if j is not None}


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