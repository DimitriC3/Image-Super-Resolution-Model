from pathlib import Path


def get_image_paths_as_dict(directory):
    """
    :param directory: (str) path to directory with dataset
    :return: (dict) key is name of image, value is path to image as string
    """

    extension = '.jpg'
    image_paths = {}

    # Create a Path object
    path = Path(directory)
    for img_path in path.rglob('*' + extension):
        image_name = img_path.stem
        # Convert each Path object to a string
        image_paths[image_name] = str(img_path)

    return image_paths
