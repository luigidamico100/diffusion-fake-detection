import matplotlib.pyplot as plt
import numpy as np
import random

from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
from defake.models.models import DnCNN, SiameseNetwork
from defake.paths import (dataset_annotations_train_path, dataset_annotations_val_path,
                          dataset_real_train_dir, dataset_real_val_dir,
                          dataset_generated_train_dir, dataset_generated_val_dir)
import pandas as pd
import seaborn as sns


def test_n1(model, dataset, n_classes, n_examples=7, device='cpu'):
    """ Script to show the distances of two random images output of the siamese network """
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    siamese_net = SiameseNetwork(model).to(device)
    
    siamese_net.eval()
    
    fig, axs = plt.subplots(n_examples, 2, figsize=(10, 10))
    dataiter = iter(dataloader)
    
    for i in range(n_examples):
      class_img_1 = random.randrange(n_classes)
      class_img_2 = random.randrange(n_classes)
      img_1 = next(dataiter)[class_img_1][0].to(device)
      img_2 = next(dataiter)[class_img_2][0].to(device)
      dist = siamese_net(img_1, img_2).item()
    
      axs[i, 0].imshow(np.transpose(img_1.cpu(), (1, 2, 0)))
      axs[i, 1].imshow(np.transpose(img_2.cpu(), (1, 2, 0)))
      axs[i, 0].set_title(f'class: {class_img_1}')
      axs[i, 1].set_title(f'class: {class_img_2}, L2-distance: {dist:.3f}')
      axs[i, 0].axis('off')
      axs[i, 1].axis('off')
    
    plt.tight_layout()


def test_histplot(model, dataset, n_classes, device='cpu'):
    """ Simple test for measuring distances of equal/different classes """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

    np.random.seed(42)

    n_examples = len(dataset) // 2
    siamese_net = SiameseNetwork(model).to(device)
    siamese_net.eval()
    dataiter = iter(dataloader)
    
    siamese_out_list = []
    for i in range(n_examples):
        class_img_1 = random.randrange(n_classes)
        class_img_2 = random.randrange(n_classes)
        img_1 = next(dataiter)[class_img_1][0].to(device) # shape: (3, H, W)
        img_2 = next(dataiter)[class_img_2][0].to(device) # shape: (3, H, W)
        dist = siamese_net(img_1, img_2).item()
        
        dict_siamese_out = {'class_img_1': class_img_1,
                            'class_img_2': class_img_2,
                            'equal_class': class_img_1==class_img_2,
                            'distance': dist}
        siamese_out_list.append(dict_siamese_out)
      
    df_siamese_out = pd.DataFrame(siamese_out_list)
    sns.histplot(df_siamese_out, x='distance', hue='equal_class', bins=50)
    
    
def test_histplot_with_model(model, dataset, n_classes, device='cpu'):
    """ Simple test for measuring distances of equal/different classes """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
    model.eval()
    
    np.random.seed(42)

    n_examples = len(dataset) // 2
    model.eval()
    dataiter = iter(dataloader)
    
    siamese_out_list = []
    
    with torch.no_grad():
        for i in range(n_examples):
            class_img_1 = random.randrange(n_classes)
            class_img_2 = random.randrange(n_classes)
            img_1 = next(dataiter)[class_img_1].to(device) # shape: (1, 3, H, W)
            img_2 = next(dataiter)[class_img_2].to(device) # shape: (1, 3, H, W)
            
            out_1 = model(img_1) # shape: (1, 1, H, W)
            out_2 = model(img_2) # shape: (1, 1, H, W)
            
            dist = torch.norm(out_1 - out_2, p=2).item()
            
            dict_siamese_out = {'class_img_1': class_img_1,
                                'class_img_2': class_img_2,
                                'equal_class': class_img_1==class_img_2,
                                'distance': dist}
            siamese_out_list.append(dict_siamese_out)
      
    df_siamese_out = pd.DataFrame(siamese_out_list)
    sns.histplot(df_siamese_out, x='distance', hue='equal_class', bins=50)
  


def test_n3(model, dataset, n_classes, n_examples=7, device='cpu'):
    """ Script to show the input and output pair of the model """
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
    
    model.eval()
    
    fig, axs = plt.subplots(n_examples, 2, figsize=(10, 10))
    dataiter = iter(dataloader)
    
    for i in range(n_examples):
        class_img = random.randrange(n_classes)
        input_img = next(dataiter)[class_img].to(device)
        output_img = model(input_img).squeeze().detach()
        input_img = input_img.squeeze()
        
        axs[i, 0].imshow(np.transpose(input_img.cpu(), (1, 2, 0)))
        axs[i, 1].imshow(np.transpose(output_img.cpu()))
        axs[i, 0].set_title(f'Input')
        axs[i, 1].set_title(f'Output')
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
    plt.tight_layout()
