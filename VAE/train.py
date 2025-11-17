import os
import numpy as np
import torch
from vposer import VPoserWrapper
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_leg_joints', type = int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--w1', type=float)
    parser.add_argument('--w2', type=float)

    args = parser.parse_args()._get_kwargs()
    kwargs = {i: j for i, j in args if j is not None}

    train_leg_poses_other_animals = torch.tensor(
    np.load(os.path.join("data", "train_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    test_leg_poses_other_animals = torch.tensor(
    np.load(os.path.join("data", "test_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    val_leg_poses_other_animals = torch.tensor(
    np.load(os.path.join("data", "val_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device =", device)

    vposer = VPoserWrapper(device=device, **kwargs)

    vposer.fit(X_train=train_leg_poses_other_animals, X_val=val_leg_poses_other_animals)

    vposer.plot_losses()