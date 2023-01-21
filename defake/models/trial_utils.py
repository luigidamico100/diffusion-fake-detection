
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        self.my_cnn = nn.Sequential(
            # nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding='same'),
        )

    def forward(self, input_):
        """
        input_ shape (n, 3, H, W)
        output_ shape (n, 1, H, W)
        """
        # input_ = input_ - .5
        output_ = self.my_cnn(input_)
        return output_


class DatasetTrial(Dataset):

    def __init__(self, dataset): # dataset: list[list[np.array]]
        """
        dataset: list[list[np.array]]
        dataset is a list containing num_class elements
         - each element is a list containing n_samples_per_class elements
         --- each element is a np.ndarray of shape (3, H, W) in the range [0..1]
        """
        self.dataset = dataset
        self.num_classes = len(self.dataset)
        
    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        """
        return a list of n_class elements
        - each element is a np.ndarray of shape (3, H, W)
        """
        return [self.dataset[class_][idx] for class_ in range(self.num_classes)]


def create_uniform_rgb(r,g,b, width = 128,height = 128):
    r_matrix = np.full((height,width), r, dtype=np.float32)
    g_matrix = np.full((height,width), g, dtype=np.float32)
    b_matrix = np.full((height,width), b, dtype=np.float32)
    # r_matrix[height-5:height-1,width-5:width-1] = 255
    # g_matrix[height-5:height-1,width-5:width-1] = 255
    # b_matrix[height-5:height-1,width-5:width-1] = 255
    return np.asanyarray([r_matrix,g_matrix,b_matrix]) # shape: (channels, height, width)

def generate_random_with_min(min=.5):
  random_number = random.random()
  return (random_number+min)/(1.+min)



def create_trial_datasets(n_samples_per_class_train=100, n_samples_per_class_val=50, n_samples_per_class_test=100):
    
    dataset_train_blue = [create_uniform_rgb(0,0, generate_random_with_min()) for _ in range(n_samples_per_class_train)]
    dataset_train_red = [create_uniform_rgb(generate_random_with_min(), 0, 0) for _ in range(n_samples_per_class_train)]
    dataset_train_green = [create_uniform_rgb(0, generate_random_with_min(), 0) for _ in range(n_samples_per_class_train)]
    dataset_val_blue = [create_uniform_rgb(0,0, generate_random_with_min()) for _ in range(n_samples_per_class_val)]
    dataset_val_red = [create_uniform_rgb(generate_random_with_min(), 0, 0) for _ in range(n_samples_per_class_val)]
    dataset_val_green = [create_uniform_rgb(0, generate_random_with_min(), 0) for _ in range(n_samples_per_class_val)]
    dataset_test_blue = [create_uniform_rgb(0,0, generate_random_with_min()) for _ in range(n_samples_per_class_test)]
    dataset_test_red = [create_uniform_rgb(generate_random_with_min(), 0, 0) for _ in range(n_samples_per_class_test)]
    dataset_test_green = [create_uniform_rgb(0, generate_random_with_min(), 0) for _ in range(n_samples_per_class_test)]
    dataset_train_list = [dataset_train_red, dataset_train_green, dataset_train_blue] # list of 3 elements -> list of images_per_classes elements -> each image has shape (3, 128, 128)
    dataset_val_list = [dataset_val_red, dataset_val_green, dataset_val_blue]
    dataset_test_list = [dataset_test_red, dataset_test_green, dataset_test_blue]
    
    dataset_train = DatasetTrial(dataset_train_list)
    dataset_val = DatasetTrial(dataset_val_list)
    dataset_test = DatasetTrial(dataset_test_list)
    n_classes = dataset_train.num_classes
    
    return dataset_train, dataset_val, dataset_test
    
    
    
    
    