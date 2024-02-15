from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


class SuperResolutionDataset(Dataset):

    def __init__(self, image_paths, scale_factor):
        """
        image_paths: list of paths to images
        scale_factor: int for downscaling factor
        """
        self.image_paths = image_paths
        self.scale_factor = scale_factor
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')

        if img.width < img.height:
            # Rotate image by 90 degrees if it's in portrait mode to make it landscape
            img = img.rotate(90, expand=True)

        orig_width, orig_height = img.size

        # Calculate low-resolution dimensions
        low_res_width, low_res_height = orig_width // self.scale_factor, orig_height // self.scale_factor

        # Create low-resolution image
        img_low_res = img.resize((low_res_width, low_res_height), Image.BICUBIC)
        img_low_res_upscaled = img_low_res.resize((orig_width, orig_height), Image.BICUBIC)

        # Use the original image as the high-resolution image
        # No need to resize since we want to keep it as the high-res target
        img_high_res = img

        # Convert to tensors
        img_low_res_tensor = self.to_tensor(img_low_res_upscaled)
        img_high_res_tensor = self.to_tensor(img_high_res)

        return img_low_res_tensor, img_high_res_tensor

