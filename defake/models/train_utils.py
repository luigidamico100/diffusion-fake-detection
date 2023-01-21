import torch
import torch.nn as nn
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


class DBLLoss():
    
    def __init__(self, batch_size, n_classes, driveaway_different_classes=False, regularization=False, lambda_=0.2, device='cpu', verbose=False):
        batch_size_per_class = batch_size // n_classes
            
        # building the correspondance matrix

        if driveaway_different_classes:
            correspondence_matrix_orig = - torch.ones(batch_size, batch_size, device=device) # N x N matrix (L in the paper)
            for idx_class in range(n_classes):
                correspondence_matrix_orig[idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class, idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class] = 1

        else:
            correspondence_matrix_orig = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device) # N x N matrix (L in the paper)
            for idx_class in range(n_classes):
                correspondence_matrix_orig[idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class, idx_class*batch_size_per_class:(idx_class+1)*batch_size_per_class] = True
        
        self.correspondence_matrix = DBLLoss.remove_diagonal(correspondence_matrix_orig) # shape: (N, N-1)
        
        self.regularization = regularization
        self.lambda_ = lambda_
        self.verbose = verbose
    
    
    def compute_loss(self, batch_outputs):
        """   
        Parameters
        ----------
        batch_outputs : torch.Tensor
            shape (N, 1, H, W)
        correspondence_matrix : torch.Tensor
            shape (N, N-1)
    
        Returns
        -------
        loss : torch.Tensor
            shape (1)
        """
        
        batch_outputs = batch_outputs.squeeze()
        
        # distance matrix
        output_vectors = torch.flatten(batch_outputs, start_dim=1, end_dim=2) # Shape (N, H*W)
        orig_distance_matrix = torch.cdist(output_vectors, output_vectors, p=2.) # Shape (N, N)
        
        # Computing softmax for each row of distance matrix - Equation (1)
        distance_matrix = DBLLoss.remove_diagonal(orig_distance_matrix) # The elements on the diagonal must not be considered in the softmax
        # Equation(1): Probability matrix    
        P = nn.functional.softmax(-distance_matrix, dim=1)
        
        # Equation (2)
        L = -torch.log((P * self.correspondence_matrix).sum(dim=1))
        
        # Equation (3)
        dbl_loss = torch.sum(L)
        
        if self.regularization:
            reg_loss = DBLLoss.compute_regularization(batch_outputs)
            reg_loss = - self.lambda_ * reg_loss
            # loss = dbl_loss - self.lambda_ * reg_loss
            
        else:
            reg_loss = 0.

        if self.verbose:
            print(f'DBL_loss: {dbl_loss:.4f}, reg_loss: {-self.lambda_ * reg_loss:.4f}')
        
        loss = dbl_loss + reg_loss
        
        return loss, dbl_loss, reg_loss
    
    @staticmethod 
    def remove_diagonal(square_matrix):
        """
        Parameters
        ----------
        square_matrix : torch.Tensor
            shape (N, N)

        Returns
        -------
        square_matrix : torch.Tensor
            shape (N, N-1)
        """
        n = len(square_matrix)
        return square_matrix.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
    
    @staticmethod
    def compute_regularization(batch_outputs):
        """
        Parameters
        ----------
        batch_outputs : torch.Tensor
            shape (N, H, W)

        Returns
        -------
        reg_loss : torch.Tensor
            shape (1)
        """
        
        def geometric_mean(input_x, dim=None):
            '''https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way'''
            log_x = torch.log(input_x)
            return torch.exp(torch.mean(log_x, dim=dim))

        # TODO: check this on doc. Maybe is the wrong one
        batch_outputs = batch_outputs.squeeze() # shape (N, H, W)
        R = torch.fft.fft2(batch_outputs, dim=(1,2)) # shape (N, H, W)
        S = torch.mean(torch.square(torch.abs(R)), dim=0) # shape (H, W)
        reg_loss = torch.log(geometric_mean(S) / torch.mean(S))
        return reg_loss
        
    
    
class PatchDataset(Dataset):
    
    def __init__(self, annotations_path, real_images_path, generated_images_path, patch_size=48, device='cpu', n_samples=None, deterministic_patches=False):
            
        with open(annotations_path) as json_file:
            annotations_dict = json.loads(json_file.read())
            self.annotations = list(annotations_dict.values())
        
        if n_samples:
            self.annotations = self.annotations[:n_samples]
        
        self.real_images_path = real_images_path
        self.generated_images_path = generated_images_path
        self.patch_size = patch_size
        self.deterministic_patches = deterministic_patches
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
        
        if self.deterministic_patches:
            real_patch_height_index = 100
            real_patch_width_index = 100
            generated_patch_height_index = 100
            generated_patch_width_index = 100
        else:
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

        
        
        return real_patch.to(self.device), generated_patch.to(self.device)
        

def compute_batch_output(model, dataloader_item):
    """
    n: n of samples per class
    N: n of total samples (N = n * n_classes)
    dataloader: list of n_class elements
    - each elements containg tensor of shape (n, 3, H, W)
    
    return batch_outputs, shape (N, 1, H, W)
    """
    batch_outputs_list = [] 
    for class_elements in dataloader_item:
        # class_elements tensor of (n, 3, 128, 128)
        # class_elements = class_elements.to(self.device)
        class_outputs = model(class_elements) #forward pass
        batch_outputs_list.append(class_outputs) # list of len=num_classes. Each elements contain a single class outputs of shape (B, 1, H, W) each
    batch_outputs = torch.concat(batch_outputs_list) # shape (N, 1, H, W) # TODO: check how it concatenate
    return batch_outputs
  
    

def main():
    
    dataset_train = PatchDataset(annotations_path=dataset_annotations_train_path,
                                 real_images_path=dataset_real_train_dir,
                                 generated_images_path=dataset_generated_train_dir)
    
    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, drop_last=True)
    
    dataloader_it = iter(dataloader_train)
    
    sample = next(dataloader_it)
    
if __name__ == '__main__':
    main()
    
  
    
  
    