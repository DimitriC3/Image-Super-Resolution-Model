import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset, DataLoader
from model.super_resolution import SuperResolutionAutoencoder

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# Hyperparameters
batch_size = 300
lr = 0.001
num_epochs = 10

criterion = nn.MSELoss()

# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SuperResolutionAutoencoder().to(device)

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

# Assuming you have your model, criterion, and test_loader defined already

transform = transforms.Compose([
    Resize((177, 217), interpolation=Image.BICUBIC),
    ToTensor()
])

# Initialize PSNR accumulator
psnr_accumulator = 0.0
num_batches = 0
test_dataset = BlurredImageDataset(root_dir_blurred='C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celba_test_blurred_2',
                                   root_dir_non_blurred='C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celba_test',
                                   transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Test loop with PSNR calculation
weights_path = 'C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\weights\\super_resolution_autoencoder_epoch_6.pth'
model.load_state_dict(torch.load(weights_path)) #load pech  2 weights
with torch.no_grad():
    for batch_idx, (blurred_imgs_test, non_blurred_imgs_test) in enumerate(test_loader):
        blurred_imgs_test, non_blurred_imgs_test = blurred_imgs_test.to(device), non_blurred_imgs_test.to(device)

        # Forward pass
        reconstructions_test = model(blurred_imgs_test)

        # Compute the loss
        loss_test = criterion(reconstructions_test, non_blurred_imgs_test)

        # Calculate PSNR
        batch_psnr = psnr(reconstructions_test, non_blurred_imgs_test)
        psnr_accumulator += batch_psnr.item()
        num_batches += 1

        # Optionally, print PSNR for each batch
        print(f"Batch {batch_idx + 1}, PSNR: {batch_psnr.item()} dB")

# Calculate average PSNR
average_psnr = psnr_accumulator / num_batches
print(f"Average PSNR: {average_psnr} dB")
# Print test loss
print(f"Test Loss: {loss_test / len(test_loader):.4f}")
