import config
import torch
from PIL import Image
import numpy as np
from skimage import exposure

class CXR_Dataset(torch.utils.data.Dataset):
    """
        Class for loading the images and their corresponding labels.
        Parameters:
        image_path (python list): A list contsisting of all the image paths (Normal and Pneunomina combined)
        transform (callable): Data augmentation pipeline.
    """
    def __init__(
        self,
        image_paths,
        transforms=None,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        try:
            # Reading the image path and the corresponding label
            img_path = self.image_paths[item]
            label = self.image_paths[item].split('/')[-2] # Splitting the string and extracting the labels (the directory name in our case)
            label = torch.tensor(config.LABEL_ENCODING[label])
            # Opening the image using Pillow and resizing it to the required size
            img = Image.open(img_path).convert("L").resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
            img = np.array(img)
            
            # performing Histogram equalization on the Image to utilize the entire pixel range
            img = exposure.equalize_hist(img)
            
            img = (((img - img.min()) / img.max() - img.min())*255).astype(np.uint8) 
            
            # Stack the Gray scaled 1 channel image 3 times to convert to 3 channel image
            img = np.stack((img, )*3) 
            img = np.transpose(img, (1, 2, 0))
            
            # Performing data augmentation using the transforms pipeline
            img = self.transforms(img)
            return {"img": img, "target": label}

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None