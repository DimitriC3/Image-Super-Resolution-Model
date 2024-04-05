import os
from PIL import Image, ImageFilter
import random

def get_image_paths(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    image_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths



def blur_images(input_dir, output_dir, blur_radius=2):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of image files in input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    # Apply blur to each image and save in the output directory
    counter = 0
    for image_file in image_files:
        if 0 == 0:
            try:
                img_path = os.path.join(input_dir, image_file)
                img = Image.open(img_path)
                blurred_img = img.filter(ImageFilter.GaussianBlur(2))
                blurred_img.save(os.path.join(output_dir, image_file))
                print(f"Blurred {image_file} and saved to {output_dir}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        else:
            try:
                img_path = os.path.join(input_dir, image_file)
                img = Image.open(img_path)
                blurred_img = img.filter(ImageFilter.BoxBlur(random.randint(0,5)))
                blurred_img.save(os.path.join(output_dir, image_file))
                print(f"Blurred {image_file} and saved to {output_dir}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        

        

#Example usage
input_directory = 'C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celba_test' 
 # Replace with path to your input directory
output_directory = 'C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celba_test_blurred_2' 
 # Replace with path to your output directory
blur_radius = 2  # Adjust the blur radius as needed

blur_images(input_directory, output_directory, blur_radius)

# Example usage
desktop_directory = 'C:\\Users\\Evan Cureton\\OneDrive\\Desktop\\img_align_celeba'  # Change this to the path of your desktop directory
train_image_paths = get_image_paths(desktop_directory)


