import matplotlib.pyplot as plt
import numpy as np
import random

from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
from torch import optim
from defake.config import device
from defake.models.models import DnCNN, SiameseNetwork, TripletModel
from defake.models.train_utils import PatchDataset, PatchDatasetTriplet
from defake.models.train_utils import DBLLoss, compute_batch_output
from defake.paths import (dataset_annotations_train_path, dataset_annotations_val_path,
                          dataset_real_train_dir, dataset_real_val_dir,
                          dataset_generated_train_dir, dataset_generated_val_dir, runs_path)
from defake.models.trial_utils import create_trial_datasets, SimpleNet
import pandas as pd
import seaborn as sns
from defake.models.test_utils import test_n1, test_histplot, test_histplot_with_model, test_n3, test_histplot_triplet
from defake.config import seed_everything
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from torch.nn import TripletMarginLoss


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


def train(batch_size, epochs, 
          regularization=True,
          lambda_=3.,
          trial=False, 
          experiment_name=None,
          load_experiment_name=None,
          perform_test=True,
          use_simplenet=False,
          device='cpu'):
    
    seed_everything(42)
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
    
    model = SimpleNet().to(device) if use_simplenet else DnCNN(in_nc=3).to(device)
        
    if load_experiment_name:
        prev_experiment_path = os.path.join(runs_path, load_experiment_name, 'model.pth')
        model.load_state_dict(torch.load(prev_experiment_path))
        print(f'Loaded model from: {load_experiment_name}')
    
    optimizer = optim.Adam(model.parameters(), lr = 0.0005)
    loss = DBLLoss(batch_size, n_classes, regularization=True, lambda_=6.5, driveaway_different_classes=False, device=device, verbose=False)
    summary(model, (3, 48, 48))
    
    # N: batch size
    # N = n_images_per_class * n_classes
    
    train_loss_history, val_loss_history = [], []
    train_loss_batches, val_loss_batches = [], []
    train_dbl_loss_batches, val_dbl_loss_batches = [], []
    train_reg_loss_batches, val_reg_loss_batches = [], []
    logs_path = os.path.join(runs_path, f"{device}_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}__{experiment_name}")
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
        print(f'train_DBL_loss: {train_dbl_loss_epoch:.5f} - train_reg_loss: {train_reg_loss_epoch:.5f}')
    
    
    writer.flush()
    writer.close()
    torch.save(model.state_dict(), os.path.join(logs_path, 'model.pth'))
    
    if perform_test:
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        test_histplot_with_model(model, dataset=dataset_train, n_classes=2, device=device, ax=axs[0])
        test_histplot_with_model(model, dataset=dataset_val, n_classes=2, device=device, ax=axs[1])
        axs[0].set_title('Train')
        axs[1].set_title('Val')
        fig.savefig(os.path.join(logs_path, 'histplots.jpg'), dpi=200)
        
    print(f'Experiment saved to: {logs_path}')
        
    
def train_triplet(batch_size, epochs, 
                  regularization=True,
                  lambda_=3.,
                  trial=False, 
                  experiment_name=None,
                  load_experiment_name=None,
                  perform_test=True,
                  use_simplenet=False,
                  device='cpu'):
    
    seed_everything(42)
    print(device)

    if not trial:
        
        n_classes = 2
        assert (batch_size/n_classes).is_integer()
        batch_size_per_class = batch_size // n_classes
        
        dataset_train = PatchDatasetTriplet(annotations_path=dataset_annotations_train_path,
                                    real_images_path=dataset_real_train_dir,
                                    generated_images_path=dataset_generated_train_dir,
                                    n_samples=None,
                                    device=device)
        
        dataset_val = PatchDatasetTriplet(annotations_path=dataset_annotations_val_path,
                                    real_images_path=dataset_real_val_dir,
                                    generated_images_path=dataset_generated_val_dir,
                                    n_samples=None,
                                    device=device)
    
    else:
        
        n_classes = 3
        assert (batch_size/n_classes).is_integer()
        
        batch_size_per_class = batch_size // n_classes
    
        dataset_train, dataset_val, dataset_test = create_trial_datasets(n_samples_per_class_train=100,
                                                                         n_samples_per_class_val=50,
                                                                         n_samples_per_class_test=500,)
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_per_class, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size_per_class, shuffle=True, drop_last=True)    
    
    model = SimpleNet() if use_simplenet else DnCNN(in_nc=3)
    triplet_model = TripletModel(model).to(device)
        
    if load_experiment_name:
        prev_experiment_path = os.path.join(runs_path, load_experiment_name, 'model.pth')
        model.load_state_dict(torch.load(prev_experiment_path))
        print(f'Loaded model from: {load_experiment_name}')
    
    optimizer = optim.Adam(model.parameters(), lr = 0.0005)
    # loss = DBLLoss(batch_size, n_classes, regularization=True, lambda_=6.5, driveaway_different_classes=False, device=device, verbose=False)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    summary(model, (3, 48, 48))
    print(triplet_model)
    
    # N: batch size
    # N = n_images_per_class * n_classes
    
    train_loss_history, val_loss_history = [], []
    train_loss_batches, val_loss_batches = [], []
    train_dbl_loss_batches, val_dbl_loss_batches = [], []
    train_reg_loss_batches, val_reg_loss_batches = [], []
    logs_path = os.path.join(runs_path, f"{device}_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}__{experiment_name}")
    writer = SummaryWriter(logs_path)
    
    
    # Iterate throught the epochs
    for epoch in range(epochs):
        train_loss_batches = []
        val_loss_batches = []
    
        triplet_model.train()
        for dataloader_item in dataloader_train:
            # dataloader_item: list of n_class elements containing tensor of shape (n, H, W)
    
            optimizer.zero_grad()
    
            # forward pass
            a_out, p_out, n_out = triplet_model(dataloader_item['a'], dataloader_item['p'], dataloader_item['n'])
    
            a_out_flatten = torch.flatten(a_out, start_dim=1, end_dim=3)
            p_out_flatten = torch.flatten(p_out, start_dim=1, end_dim=3)
            n_out_flatten = torch.flatten(n_out, start_dim=1, end_dim=3)

            batch_loss = triplet_loss(a_out_flatten, p_out_flatten, n_out_flatten)
    
            batch_loss.backward()
            optimizer.step()
    
            train_loss_batches.append(batch_loss.item())
            
            print('-', end='')
    
        triplet_model.eval()
        with torch.no_grad():
            for dataloader_item in dataloader_val:
                # batch_outputs = model.compute_batch_output(dataloader_item)
                a_out, p_out, n_out = triplet_model(dataloader_item['a'], dataloader_item['p'], dataloader_item['n'])
                batch_loss = triplet_loss(a_out_flatten, p_out_flatten, n_out_flatten)
                val_loss_batches.append(batch_loss.item())

        # update the training history 
        train_loss_epoch = sum(train_loss_batches) / len(train_loss_batches)
        val_loss_epoch = sum(val_loss_batches) / len(val_loss_batches)
        
        # test_loss_epoch = sum(test_loss_batches) / len(test_loss_batches)
        train_loss_history.append(train_loss_epoch)
        val_loss_history.append(val_loss_epoch)
        # test_loss_history.append(test_loss_epoch)
        writer.add_scalar("Loss/train", train_loss_epoch, epoch)
        writer.add_scalar("Loss/val", val_loss_epoch, epoch)
    
        # if epoch%4 == 0:
        # print(f'\nEpoch: {epoch} - train_loss: {train_loss_epoch:.5f} - val_loss: {val_loss_epoch:.5f} - test_loss: {test_loss_epoch:.5}') 
        print(f'\nEpoch: {epoch} - train_loss: {train_loss_epoch:.5f} - val_loss: {val_loss_epoch:.5f}')
    
    
    writer.flush()
    writer.close()
    torch.save(model.state_dict(), os.path.join(logs_path, 'model.pth'))
    
    if perform_test:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        test_histplot_triplet(model, dataset=dataset_train, device=device, ax=axs[0])
        test_histplot_triplet(model, dataset=dataset_val, device=device, ax=axs[1])
        axs[0].set_title('Train')
        axs[1].set_title('Val')
        fig.savefig(os.path.join(logs_path, 'histplots.jpg'), dpi=200)
        
    print(f'Experiment saved to: {logs_path}')
        
    
    
    
#%%


if __name__ == '__main__':
    train_triplet(batch_size=128, 
            epochs=40, 
            regularization=True,
            lambda_=3,
            trial=False, 
            experiment_name=None, 
            load_experiment_name=None,
            perform_test=True,
            use_simplenet=True,
            device='cpu')









