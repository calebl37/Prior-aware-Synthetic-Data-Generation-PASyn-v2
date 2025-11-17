import os
import numpy as np
import torch
from vposer import VPoserWrapper



train_leg_poses_other_animals = torch.tensor(
    np.load(os.path.join("data", "train_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

test_leg_poses_other_animals = torch.tensor(
    np.load(os.path.join("data", "test_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

val_leg_poses_other_animals = torch.tensor(
    np.load(os.path.join("data", "val_poses_refine.npz"))["poses"][:, 3:], dtype=torch.float32)

# print(train_leg_poses_other_animals.shape)
# print(test_leg_poses_other_animals.shape)
# print(val_leg_poses_other_animals.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device =", device)

vposer = VPoserWrapper(device=device, epochs=250, batch_size=128, w1=0.005, w2=0.01)

vposer.fit(X_train=train_leg_poses_other_animals, X_val=val_leg_poses_other_animals)

vposer.plot_losses()