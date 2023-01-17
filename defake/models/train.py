import matplotlib.pyplot as plt
import numpy as np
import random

from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
from torch import optim
from defake.config import device
from defake.models.models import DnCNN, SiameseNetwork
from defake.models.data_manager import PatchDataset
from defake.models.train_utils import DBLLoss, compute_batch_output
from defake.paths import (dataset_annotations_train_path, dataset_annotations_val_path,
                          dataset_real_train_dir, dataset_real_val_dir,
                          dataset_generated_train_dir, dataset_generated_val_dir)
from defake.models.trial_utils import create_trial_datasets, SimpleNet
import pandas as pd
import seaborn as sns
from defake.models.test_utils import test_n1, test_histplot, test_histplot_with_model, test_n3
from defake.config import seed_everything

seed_everything(42)

device = torch.device("cpu")
print(device)

#%% Params

batch_size = 60 # Choose batch_size > n_classes*3
epochs = 5
trial = True

#%% Initialization

if not trial:
    
    n_classes = 2
    assert (batch_size/n_classes).is_integer()
    batch_size_per_class = batch_size // n_classes
    
    dataset_train = PatchDataset(annotations_path=dataset_annotations_train_path,
                                real_images_path=dataset_real_train_dir,
                                generated_images_path=dataset_generated_train_dir)
    
    dataset_val = PatchDataset(annotations_path=dataset_annotations_val_path,
                                real_images_path=dataset_real_val_dir,
                                generated_images_path=dataset_generated_val_dir)
    dataset_test = dataset_val

else:
    
    n_classes = 3
    assert (batch_size/n_classes).is_integer()
    
    batch_size_per_class = batch_size // n_classes

    dataset_train, dataset_val, dataset_test = create_trial_datasets(n_samples_per_class_train=100,
                                                                     n_samples_per_class_val=50,
                                                                     n_samples_per_class_test=500,)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size_per_class, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size_per_class, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, drop_last=True)



# model = DnCNN(in_nc=3) # This model is really huge -> the training gonna be slow
model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.0005)


#%% Training loop

# N: batch size
# N = n_images_per_class * n_classes

train_loss_history, val_loss_history = [], []
train_loss_batches, val_loss_batches = [], []

loss = DBLLoss(batch_size, n_classes, device)

# Iterate throught the epochs
for epoch in range(epochs):
    train_loss_batches = []
    val_loss_batches = []

    model.train()
    for dataloader_item in dataloader_train:
        # dataloader_item: list of n_class elements containing tensor of shape (n, H, W)

        optimizer.zero_grad()

        # forward pass
        # batch_outputs = model.compute_batch_output(dataloader_item)
        batch_outputs = compute_batch_output(model, dataloader_item)

        batch_loss = loss.compute_loss(batch_outputs)

        batch_loss.backward()
        optimizer.step()

        train_loss_batches.append(batch_loss.item())
        
        print('-', end='')

    model.eval()
    with torch.no_grad():
        for dataloader_item in dataloader_val:
            # batch_outputs = model.compute_batch_output(dataloader_item)
            batch_outputs = compute_batch_output(model, dataloader_item)
            batch_loss = loss.compute_loss(batch_outputs)
            val_loss_batches.append(batch_loss.item())

    # update the training history 
    train_loss_epoch = sum(train_loss_batches) / len(train_loss_batches)
    val_loss_epoch = sum(val_loss_batches) / len(val_loss_batches)
    train_loss_history.append(train_loss_epoch)
    val_loss_history.append(val_loss_epoch)

    # if epoch%4 == 0:
    print(f'\nEpoch: {epoch} - train_loss: {train_loss_epoch:.10f} - val_loss: {val_loss_epoch:.10f}') 

plt.plot(train_loss_history, label='training loss')
plt.plot(val_loss_history, label='validation loss')
plt.legend()



#%% Test

dataset = dataset_train
n_examples = 7

test_n1(model, dataset, n_classes, n_examples, device)
  
test_histplot_with_model(model, dataset, n_classes, device)
test_histplot(model, dataset, n_classes, device)

test_n3(model, dataset, n_classes, n_examples, device)











