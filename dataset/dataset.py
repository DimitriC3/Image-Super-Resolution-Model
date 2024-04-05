import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
from torchvision import transforms
from torchvision.utils import save_image
from model.super_resolution import SuperResolutionAutoencoder  # Assuming you have defined your model in a separate file called model.py
from PIL import Image


class BlurredImageDataset(Dataset):
    def __init__(self, root_dir_blurred, root_dir_non_blurred, transform=None):
        self.root_dir_blurred = root_dir_blurred
        self.root_dir_non_blurred = root_dir_non_blurred
        self.blurred_images = os.listdir(root_dir_blurred)
        self.non_blurred_images = os.listdir(root_dir_non_blurred)
        self.transform = transform

    def __len__(self):
        return min(len(self.blurred_images), len(self.non_blurred_images))

    def __getitem__(self, idx):
        blurred_img_name = os.path.join(self.root_dir_blurred, self.blurred_images[idx])
        non_blurred_img_name = os.path.join(self.root_dir_non_blurred, self.non_blurred_images[idx])
        
        blurred_img = Image.open(blurred_img_name).convert('RGB')
        non_blurred_img = Image.open(non_blurred_img_name).convert('RGB')

        if self.transform:
            blurred_img = self.transform(blurred_img)
            non_blurred_img = self.transform(non_blurred_img)

        return blurred_img, non_blurred_img
        
