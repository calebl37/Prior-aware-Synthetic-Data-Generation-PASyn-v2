import numpy as np
from vposer import VPoserWrapper
import argparse
import torch
import os
import math



if __name__ == "__main__":

    

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_frames', type=int)

    train_leg_poses_other_animals = torch.tensor(
        np.load(os.path.join("data", "train_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    test_leg_poses_other_animals = torch.tensor(
        np.load(os.path.join("data", "test_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    val_leg_poses_other_animals = torch.tensor(
        np.load(os.path.join("data", "val_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    args = parser.parse_args()

    n_frames = args.n_frames

    full_data = torch.cat([train_leg_poses_other_animals, test_leg_poses_other_animals, val_leg_poses_other_animals], dim=0)

    n_repeats = math.ceil(n_frames / full_data.shape[0])
    if n_repeats > 1:
        full_data = full_data.repeat(n_repeats,1)

    source = full_data[:n_frames]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vposer = VPoserWrapper(device=device)

    generated_leg_poses = vposer.predict(source)

    root = os.path.dirname(os.getcwd())

    np.save(os.path.join(root, "blender", "my_vae_poses.npy"), generated_leg_poses)