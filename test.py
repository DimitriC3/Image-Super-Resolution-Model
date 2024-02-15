# Get image paths for images, can use function in dataset.utils
from LowerResolution import test_image_paths

# From GitHub
from dataset.dataset import SuperResolutionDataset
from model.super_resolution import SuperResolutionAutoencoder
from train import path  # path to saved model

# Python Libraries
import torch
from torch.utils.data import DataLoader


test_image_paths = list(test_image_paths.values())
test_dataset = SuperResolutionDataset(image_paths=test_image_paths, scale_factor=4)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperResolutionAutoencoder().to(device)

# Input your own path to saved model, imported from train.py
model.load_state_dict(torch.load(path))
model.eval()
model.to(device)


# Function to calculate PSNR, assuming max pixel value is 1.0
def psnr(target, prediction):
    mse = ((target - prediction) ** 2).mean().item()
    return 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse)))


# Loop over the test set to evaluate the model
total_psnr = 0.0
with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        # Calculate PSNR for each image and accumulate
        total_psnr += psnr(targets, outputs)

# Calculate the average PSNR over all test images
avg_psnr = total_psnr / len(test_dataloader)
print(f'Average PSNR on the test set: {avg_psnr:.2f} dB')
