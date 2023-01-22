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
from defake.models.train_utils import PatchDataset
from defake.models.train_utils import DBLLoss, compute_batch_output
from defake.paths import (dataset_annotations_train_path, dataset_annotations_val_path,
                          dataset_real_train_dir, dataset_real_val_dir,
                          dataset_generated_train_dir, dataset_generated_val_dir, runs_path)
from defake.models.trial_utils import create_trial_datasets, SimpleNet
import pandas as pd
import seaborn as sns
from defake.models.test_utils import test_n1, test_histplot, test_histplot_with_model, test_n3
from defake.config import seed_everything
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


'''
- DataLoader has bad default settings, tune num workers > 0 and default to pin memory = True
- use torch.backends. cudnn. benchmark = True to autotune cudnn kernel choice
- max out the batch size for each GPU to ammortize compute
- do not forget bias=False in weight layers before BatchNorms, it's a noop that bloats model
- use for p in model.parameters ( ): p.grad = None instead of model.zero grad ( )
- careful to disable debug APIs in prod (detect _anomaly/profiler/emit _nvtx/gradcheck...)
- use DistributedDataParallel not DataParallel, even if not running distributed 
- careful to load balance compute on all GPUs if variably-sized inputs or GPUs will idle 
- use an apex fused optimizer (default PyTorch optim for loop iterates individual params, yikes) 
- use checkpointing to recompute memory-intensive compute-efficient ops in bwd pass (eg activations, upsampling,
- use @torch.jit.script, e.g. esp to fuse long sequences of pointwise ops like in GELU
'''


def train(batch_size, epochs, trial=False):
    
    seed_everything(42)
    device = torch.device("cpu")
    print(device)

    if not trial:
        
        n_classes = 2
        assert (batch_size/n_classes).is_integer()
        batch_size_per_class = batch_size // n_classes
        
        dataset_train = PatchDataset(annotations_path=dataset_annotations_train_path,
                                    real_images_path=dataset_real_train_dir,
                                    generated_images_path=dataset_generated_train_dir,
                                    n_samples=None,
                                    deterministic_patches=False,
                                    device=device)
        
        dataset_val = PatchDataset(annotations_path=dataset_annotations_val_path,
                                    real_images_path=dataset_real_val_dir,
                                    generated_images_path=dataset_generated_val_dir,
                                    n_samples=None,
                                    device=device)
        dataset_test = dataset_val
    
    else:
        
        n_classes = 3
        assert (batch_size/n_classes).is_integer()
        
        batch_size_per_class = batch_size // n_classes
    
        dataset_train, dataset_val, dataset_test = create_trial_datasets(n_samples_per_class_train=100,
                                                                         n_samples_per_class_val=50,
                                                                         n_samples_per_class_test=500,)
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_per_class, shuffle=False, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size_per_class, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, drop_last=True)
    
    
    
    model = DnCNN(in_nc=3).to(device)
    # model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0005)
    loss = DBLLoss(batch_size, n_classes, regularization=True, lambda_=10, driveaway_different_classes=False, device=device, verbose=False)
    summary(model, (3, 48, 48))
    
    # N: batch size
    # N = n_images_per_class * n_classes
    
    train_loss_history, val_loss_history, test_loss_history = [], [], []
    train_loss_batches, val_loss_batches, test_loss_batches = [], [], []
    train_dbl_loss_batches, val_dbl_loss_batches, test_dbl_loss_batches = [], [], []
    train_reg_loss_batches, val_reg_loss_batches, test_reg_loss_batches = [], [], []
    logs_path = os.path.join(runs_path, datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    writer = SummaryWriter(logs_path)
    
    
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
    
            batch_loss, batch_dbl_loss, batch_reg_loss = loss.compute_loss(batch_outputs)
    
            batch_loss.backward()
            optimizer.step()
    
            train_loss_batches.append(batch_loss.item())
            train_dbl_loss_batches.append(batch_dbl_loss.item())
            train_reg_loss_batches.append(batch_reg_loss.item())
            
            print('-', end='')
    
        model.eval()
        with torch.no_grad():
            for dataloader_item in dataloader_val:
                # batch_outputs = model.compute_batch_output(dataloader_item)
                batch_outputs = compute_batch_output(model, dataloader_item)
                batch_loss, batch_dbl_loss, batch_reg_loss = loss.compute_loss(batch_outputs)
                val_loss_batches.append(batch_loss.item())
                val_dbl_loss_batches.append(batch_dbl_loss.item())
                val_reg_loss_batches.append(batch_reg_loss.item())
                
        # # test ??
        # model.train()
        # # with torch.no_grad():
        # for dataloader_item in dataloader_train:
        #     # batch_outputs = model.compute_batch_output(dataloader_item)
        #     batch_outputs = compute_batch_output(model, dataloader_item)
        #     batch_loss, batch_dbl_loss, batch_reg_loss = loss.compute_loss(batch_outputs)
        #     test_loss_batches.append(batch_loss.item())
    
        # update the training history 
        train_loss_epoch = sum(train_loss_batches) / len(train_loss_batches)
        val_loss_epoch = sum(val_loss_batches) / len(val_loss_batches)
        
        train_dbl_loss_epoch = sum(train_dbl_loss_batches) / len(train_dbl_loss_batches)
        val_dbl_loss_epoch = sum(val_dbl_loss_batches) / len(val_dbl_loss_batches)
        train_reg_loss_epoch = sum(train_reg_loss_batches) / len(train_reg_loss_batches)
        val_reg_loss_epoch = sum(val_reg_loss_batches) / len(val_reg_loss_batches)
        
        # test_loss_epoch = sum(test_loss_batches) / len(test_loss_batches)
        train_loss_history.append(train_loss_epoch)
        val_loss_history.append(val_loss_epoch)
        # test_loss_history.append(test_loss_epoch)
        writer.add_scalar("Loss/train", train_loss_epoch, epoch)
        writer.add_scalar("DBLLoss/train", train_dbl_loss_epoch, epoch)
        writer.add_scalar("RegLoss/train", train_reg_loss_epoch, epoch)
        writer.add_scalar("Loss/val", val_loss_epoch, epoch)
        writer.add_scalar("DBLLoss/val", val_dbl_loss_epoch, epoch)
        writer.add_scalar("RegLoss/val", val_reg_loss_epoch, epoch)
    
        # if epoch%4 == 0:
        # print(f'\nEpoch: {epoch} - train_loss: {train_loss_epoch:.5f} - val_loss: {val_loss_epoch:.5f} - test_loss: {test_loss_epoch:.5}') 
        print(f'\nEpoch: {epoch} - train_loss: {train_loss_epoch:.5f} - val_loss: {val_loss_epoch:.5f}') 
    
    
    writer.flush()
    writer.close()
    torch.save(model.state_dict, os.path.join(logs_path, 'model'))
    print(f'Logs and model saved to: {logs_path}')
    
    # plt.plot(train_loss_history, label='training loss')
    # plt.plot(val_loss_history, label='validation loss')
    # plt.legend()
    


# #%% compute loss on all the dataset_train


# model.train()
# with torch.no_grad():
#     for dataloader_item in dataloader_train:
#         # batch_outputs = model.compute_batch_output(dataloader_item)
#         batch_outputs = compute_batch_output(model, dataloader_item)
#         batch_loss = loss.compute_loss(batch_outputs)
#         test_loss_batches.append(batch_loss.item())
#         print('-', end='')
        
# print()
# test_loss_epoch = sum(test_loss_batches) / len(test_loss_batches)
# print(test_loss_epoch)

        



def test():

    #%% test
    
    dataset = dataset_train
    test_histplot_with_model(model, dataset, n_classes, device)
    
    #%% Test
    
    dataset = dataset_train
    n_examples = 7
    
    test_n1(model, dataset, n_classes, n_examples, device)
      
    test_histplot_with_model(model, dataset, n_classes, device)
    test_histplot(model, dataset, n_classes, device)
    
    test_n3(model, dataset, n_classes, n_examples, device)



if __name__ == '__main__':
    train(batch_size=10,
          epochs=10,
          trial=False)









