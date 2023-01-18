from torch.utils.data import DataLoader, Dataset
from defake.paths import (dataset_annotations_train_path, dataset_annotations_val_path,
                          dataset_real_train_dir, dataset_real_val_dir,
                          dataset_generated_train_dir, dataset_generated_val_dir)
import json
from PIL import Image
import os
import torchvision.transforms.functional as TF
from numpy.random import randint
import torch


class PatchDataset(Dataset):
    
    def __init__(self, annotations_path, real_images_path, generated_images_path, patch_size=48, device='cpu'):
            
        with open(annotations_path) as json_file:
            annotations_dict = json.loads(json_file.read())
            self.annotations = list(annotations_dict.values())
        
        self.real_images_path = real_images_path
        self.generated_images_path = generated_images_path
        self.patch_size = patch_size
        self.device = device
            
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        return a list of n_class elements
        - each element is a np.ndarray of shape (3, H, W)
        """
        
        real_image_name = self.annotations[idx]['image_name']
        generated_image_name = self.annotations[idx]['generated_image_name']
        prompt = self.annotations[idx]['prompt']
        
        real_image_path = os.path.join(self.real_images_path, real_image_name)
        generated_image_path = os.path.join(self.generated_images_path, generated_image_name)
        
        real_image = TF.to_tensor(Image.open(real_image_path))
        generated_image = TF.to_tensor(Image.open(generated_image_path))
        
        real_patch_height_index = randint(0, real_image.shape[1] - self.patch_size)
        real_patch_width_index = randint(0, real_image.shape[2] - self.patch_size)
        generated_patch_height_index = randint(0, generated_image.shape[1] - self.patch_size)
        generated_patch_width_index = randint(0, generated_image.shape[2] - self.patch_size)
        
        real_patch = real_image[:, 
                                real_patch_height_index:real_patch_height_index+self.patch_size, 
                                real_patch_width_index:real_patch_width_index+self.patch_size]
        
        if len(real_patch) == 1:
            # This image is not RGB
            real_patch = torch.cat([real_patch, real_patch, real_patch], 0)
        
        generated_patch = generated_image[:, 
                                          generated_patch_height_index:generated_patch_height_index+self.patch_size, 
                                          generated_patch_width_index:generated_patch_width_index+self.patch_size]

        
        
        return [real_patch.to(self.device), generated_patch.to(self.device)]
        
    
    
def main():
    
    dataset_train = PatchDataset(annotations_path=dataset_annotations_train_path,
                                 real_images_path=dataset_real_train_dir,
                                 generated_images_path=dataset_generated_train_dir)
    
    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, drop_last=True)
    
    dataloader_it = iter(dataloader_train)
    
    sample = next(dataloader_it)
    
if __name__ == '__main__':
    main()
        
            
        

    

