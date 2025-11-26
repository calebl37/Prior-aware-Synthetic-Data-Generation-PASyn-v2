import numpy as np
from vposer import VPoserWrapper
import argparse
import torch
import os
import math



if __name__ == "__main__":

    #get number of desired frames from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_frames', type=int, default=3000)
    args = parser.parse_args()
    n_frames = args.n_frames

    #load 3 datasets - the leg poses from other low poly animated 3D animal models
    train_leg_poses_other_animals = torch.tensor(
        np.load(os.path.join("data", "train_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)
    test_leg_poses_other_animals = torch.tensor(
        np.load(os.path.join("data", "test_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)
    val_leg_poses_other_animals = torch.tensor(
        np.load(os.path.join("data", "val_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    #concatenate the 3 datasets together
    full_data = torch.cat([train_leg_poses_other_animals, test_leg_poses_other_animals, val_leg_poses_other_animals], dim=0)

    #tile the the 3 datasets until shape[0] >= n_frames
    n_repeats = math.ceil(n_frames / full_data.shape[0])
    if n_repeats > 1:
        full_data = full_data.repeat(n_repeats,1)

    #clip the tiled dataset to shape (n_frames, n_leg_joints*3)
    source = full_data[:n_frames]

    #GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #initialize the VAE (no training needed)
    vposer = VPoserWrapper(device=device)

    #generate leg poses for n_frames frames, using the current weights of the VAE
    generated_leg_poses = vposer.predict(source)

    #save output to the directory with the blender script
    root = os.path.dirname(os.getcwd())
    np.save(os.path.join(root, "blender", "my_vae_poses.npy"), generated_leg_poses.cpu())