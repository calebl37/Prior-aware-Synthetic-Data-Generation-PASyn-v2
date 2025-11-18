import torch
import torchvision.datasets as dset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import numpy as np
import math


def rgba_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGBA")




#save output to the directory with the blender script
synthetic_dataset = dset.ImageFolder(root=os.path.join("data", "synthetic_images"),
                                     loader=rgba_loader,
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.CenterCrop((512,512))]))

fake_zebra_images = torch.stack([t[0][:3, :, :] for t in synthetic_dataset], dim = 0)
fake_zebra_alphas = torch.stack([t[0][3, :, :] for t in synthetic_dataset], dim = 0).unsqueeze(dim=1)


background_dataset = dset.ImageFolder(root=os.path.join("data", "backgrounds"),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Resize((512,512))]))

real_backgrounds = torch.stack([t[0] for t in background_dataset], dim = 0)

n_repeats = math.ceil(real_backgrounds.shape[0] / fake_zebra_images.shape[0])
if n_repeats > 1:
    real_backgrounds = fake_zebra_images.repeat(n_repeats,1, 1, 1)
real_backgrounds = fake_zebra_images[:fake_zebra_images.shape[0]]

print(fake_zebra_images.shape)
print(fake_zebra_alphas.shape)
print(real_backgrounds.shape)