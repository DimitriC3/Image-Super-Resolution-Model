import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
from torchvision import transforms
from torchvision.utils import save_image
from model.super_resolution import SuperResolutionAutoencoder  # Assuming you have defined your model in a separate file called model.py
from PIL import Image

#weight directory
weights_dir = "G:\\My Drive\\weights"
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

# Define a custom dataset class
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



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 16
lr = 0.001
num_epochs = 10

# Initialize the model
model = SuperResolutionAutoencoder().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Prepare dataset and dataloaders
transform = transforms.Compose([
    Resize((177, 217), interpolation=Image.BICUBIC),
    ToTensor()
])
train_dataset = BlurredImageDataset(root_dir_blurred='C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celeba_training_small',
                                    root_dir_non_blurred='C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celeba_small',
                                    transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Prepare validation and test datasets
validation_dataset = BlurredImageDataset(root_dir_blurred='C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celeba_valid_blurred',
                                         root_dir_non_blurred='C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celeba_valid',
                                         transform=transform)

test_dataset = BlurredImageDataset(root_dir_blurred='C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celba_test_blurred',
                                   root_dir_non_blurred='C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celba_test',
                                   transform=transform)

# Create validation and test data loaders
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
# weights_path = 'G:\\My Drive\\weights\\new_super_resolution_autoencoder_epoch_10.pth'
# model.load_state_dict(torch.load(weights_path)) #load pech  2 weights

for epoch in range(num_epochs):
    print("new epoch loading...")
    model.train()
    running_loss = 0.0
    for batch_idx, (blurred_imgs, non_blurred_imgs) in enumerate(train_loader):
        blurred_imgs, non_blurred_imgs = blurred_imgs.to(device), non_blurred_imgs.to(device)

        # Forward pass
        reconstructions = model(blurred_imgs)

        # Compute the loss
        loss = criterion(reconstructions, non_blurred_imgs)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    

    # Print average loss for the epoch
    weights_path = os.path.join(weights_dir, f"3news_super_resolution_autoencoder_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights at epoch {epoch+1} to {weights_path}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    if epoch % 2 == 0:


        # Save the trained model
        torch.save(model.state_dict(), 'super_resolution_autoencoder.pth')

        # Validation loop
        model.eval()  # Set model to evaluation mode
        validation_loss = 0.0
        with torch.no_grad():
            for batch_idx, (blurred_imgs_val, non_blurred_imgs_val) in enumerate(validation_loader):
                blurred_imgs_val, non_blurred_imgs_val = blurred_imgs_val.to(device), non_blurred_imgs_val.to(device)
                
                # Forward pass
                reconstructions_val = model(blurred_imgs_val)
                
                # Compute the loss
                loss_val = criterion(reconstructions_val, non_blurred_imgs_val)
                
                validation_loss += loss_val.item()
                
                
      
            
        # Print validation loss
        print(f"Validation Loss: {validation_loss / len(validation_loader):.4f}")

        # Test loop
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (blurred_imgs_test, non_blurred_imgs_test) in enumerate(test_loader):
            blurred_imgs_test, non_blurred_imgs_test = blurred_imgs_test.to(device), non_blurred_imgs_test.to(device)
            
            # Forward pass
            reconstructions_test = model(blurred_imgs_test)
            
            # Compute the loss
            loss_test = criterion(reconstructions_test, non_blurred_imgs_test)
            
            
            
            

    # Calculate the average PSNR over all test images
        
    # Print test loss
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
