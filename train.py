# Get image paths for images, can use function in dataset.utils
from LowerResolution import train_image_paths

# From GitHub
from dataset.dataset import SuperResolutionDataset
from model.super_resolution import SuperResolutionAutoencoder

# Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


train_image_paths = list(train_image_paths.values())
dataset = SuperResolutionDataset(image_paths=train_image_paths, scale_factor=4)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SuperResolutionAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Ensure the shapes match before calculating loss
        if outputs.shape != targets.shape:
            print(f'Output shape: {outputs.shape}')
            print(f'Target shape: {targets.shape}')
            print("Shape mismatch detected!")
            continue

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print statistics
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')

print('Finished Training')

# Save model (optional)

answer = input("Do you want to save the model? (y/n): ")
path = ''
if answer == 'y':
    path = input("Input path to save location ending with .pth (i.e. /.../.../file_name.pth")
    model_state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(model.state_dict(), path)
else:
    print("Did not save")

