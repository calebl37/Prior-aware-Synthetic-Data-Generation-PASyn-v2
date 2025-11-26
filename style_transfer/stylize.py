import numpy as np
import torch
import torchvision.datasets as dset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import argparse
import math
import matplotlib.pyplot as plt
from cnn_adain_model import ConvStyleTransfer


def rgba_loader(path: str) -> Image:
    '''
    Reads an PNG image file and converts to RGBA. 
    The A (alpha) channel creates a binary PNG map where 0 = transparent and 1 = opaque

    Args:
        path (str): the filename of the image
    
    Returns:
        RGBA PIL image
    '''
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGBA")


if __name__ == "__main__":

    #get the sdimensions of the image and the blending strength parameter from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()
    image_height = args.height
    image_width = args.width
    alpha = args.alpha

    print("Loading Blender Generated Zebra Images...")
    #load blender zebra images as RGBA
    synthetic_dataset = dset.ImageFolder(root=os.path.join("data", "synthetic_images"),
                                        loader=rgba_loader,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Resize((image_height,image_width))]))

    #separate the RGB data from the alpha channel (PNG map 0 for transparent, 1 for opaque)
    fake_zebra_images = torch.stack([t[0][:3, :, :] for t in synthetic_dataset], dim = 0)
    fake_zebra_alphas = torch.stack([t[0][3, :, :] for t in synthetic_dataset], dim = 0).unsqueeze(dim=1)
    print("...Done")


    print("Loading real background images...")
    #load the real backgrounds as RGB
    background_dataset = dset.ImageFolder(root=os.path.join("data", "backgrounds"),
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.2, 0.4), ratio=(0.75, 1.33))]))

    real_backgrounds = torch.stack([t[0] for t in background_dataset], dim = 0)

    #repeat the backgrounds until there is one background for every blender zebra image
    n_repeats = math.ceil(real_backgrounds.shape[0] / fake_zebra_images.shape[0])
    if n_repeats > 1:
        real_backgrounds = real_backgrounds.repeat(n_repeats,1, 1, 1)
    real_backgrounds = real_backgrounds[:fake_zebra_images.shape[0]]

    print("...done")

    #GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load instance of the CNN+AdAIN style transfer model
    cnn_adain_model = ConvStyleTransfer(device=device, height=image_height, width=image_width)

    fake_zebra_alphas= fake_zebra_alphas.to(device)
    fake_zebra_images = fake_zebra_images.to(device)

    stylized_zebras = cnn_adain_model.predict(content=fake_zebra_images, style=real_backgrounds, alpha=alpha)
    composite = fake_zebra_alphas * stylized_zebras + (1 - fake_zebra_alphas) * real_backgrounds


    #save tensor format
    torch.save(composite, 'stylized_images.pt')
    

    #show a few of them
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Stylized Zebras")
    plt.imshow(np.transpose(vutils.make_grid(composite[:16], nrow=4, padding=1, normalize=False).cpu(),(1,2,0)))
    plt.show()

    #save to image folder
    if not os.path.exists("stylized_images"):
        os.mkdir("stylized_images")
    for i in range(composite.shape[0]):
        vutils.save_image(composite[i], os.path.join("stylized_images", "stylized_zebra_{}.jpg".format(i)))
    





