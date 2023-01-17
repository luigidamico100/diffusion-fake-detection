import matplotlib.pyplot as plt
import numpy as np
import random

from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
from torch import optim
from defake.config import device


class SiameseNetwork(nn.Module):
    def __init__(self, model):
      super(SiameseNetwork, self).__init__()
      self.model = model

    def forward(self, input1, input2):
      """
      DO NOT use batch! 
      input1: shape (3, H, W)
      input2: shape (3, H, W)
      """
      assert len(input1.shape)==3 and len(input2.shape)==3, "You have used batch!"
      
      input1 = input1.unsqueeze(dim=0) # shape (1, 3, H, W)
      input2 = input2.unsqueeze(dim=0) # shape (1, 3, H, W)

      output1 = self.model(input1) # shape (1, 1, H, W)
      output2 = self.model(input2) # shape (1, 1, H, W)
      output1_flatten = torch.flatten(output1)
      output2_flatten = torch.flatten(output2)

      return torch.norm(output1_flatten - output2_flatten, p=2)


class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, pretrained_path = None, device = 'cpu'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU
        pretrained_path: if not None, load pretrained
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super().__init__()
        
        bias = True

        m_head = [
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        ]
        
        m_body = [layer for levels in [[
            nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=bias),
            # nn.BatchNorm2d(num_features=nc, momentum=0.9, eps=1e-04, affine=True), # TODO: understand why!
            nn.ReLU(inplace=True)
        ] for _ in range(nb-2)] for layer in levels]
        
        m_tail = [
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=bias)
        ]

        self.model = nn.Sequential(*m_head, *m_body, *m_tail)
        
        if pretrained_path is not None:
            self.load_weights(pretrained_path, device = device)

    ''' -------------------- training -------------------- '''

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2) # (batch, patch_x, patch_y, channel) -> (batch, channel, patch_x, patch_y)
        n = self.model(x)
        # n = n.permute(0, 2, 3, 1) # (batch, channel, patch_x, patch_y) -> (batch, patch_x, patch_y, channel)
        return n # we need only the noise (n), not the denoised image (x-n)
        
    ''' -------------------- prediction -------------------- '''
    
    def model_predict_patch(self, img):
        return self(
            torch.from_numpy(img).to(
                next(self.parameters()).device
            )
        )

    def model_predict_small(self, img):
        return np.squeeze(self.model_predict_patch(img[np.newaxis, :, :, np.newaxis]).detach().cpu().numpy())

    def model_predict_large(self, img):
        slide = 1024  # 3072
        overlap = 34
        # prepare output array
        res = np.zeros((img.shape[0], img.shape[1]), np.float32)

        # iterate over x and y, strides = slide, window size = slide+2*overlap
        for x in range(0, img.shape[0], slide):
            x_start = x - overlap
            x_end = x + slide + overlap
            for y in range(0, img.shape[1], slide):
                y_start = y - overlap
                y_end = y + slide + overlap
                patch = img[max(x_start, 0): min(x_end, img.shape[0]), max(y_start, 0): min(y_end, img.shape[1])]
                patch_res = np.squeeze(self.model_predict_small(patch))

                # discard initial overlap if not the row or first column
                if x > 0:
                    patch_res = patch_res[overlap:, :]
                if y > 0:
                    patch_res = patch_res[:, overlap:]
                # discard data beyond image size
                patch_res = patch_res[:min(slide, patch_res.shape[0]), :min(slide, patch_res.shape[1])]
                # copy data to output buffer
                res[x: min(x + slide, res.shape[0]), y: min(y + slide, res.shape[1])] = patch_res
        return res

    def predict(self, img):
        '''
        Run the diffusionprint generation CNN over the input image
        :param img: input image, 2-D numpy array
        :return: output diffusionprint, 2-D numpy array with the same size of the input image
        '''
        large_limit = 1050000  # 9437184
        if len(img.shape) != 2:
            raise ValueError("Input image must be 2-dimensional. Passed shape: %r" % img.shape)
        if img.shape[0] * img.shape[1] > large_limit:
            res = self.model_predict_large(img)
        else:
            res = self.model_predict_small(img)
            
        return self.normalize_diffusionprint(res)
    
    def predict_file(self, path):
        ''' Open WxHx3 image and convert to [0,1) interval of dimension WxH, then computes the model '''
        img = np.asarray(Image.open(path).convert("YCbCr"))[..., 0].astype(np.float32) / 256.0
        return self.predict(img)
        
    def normalize_diffusionprint(self, diffusionprint, margin=34):
        '''
        Normalize the diffusionprint between 0 and 1, in respect to the central area
        :param diffusionprint: diffusionprint data, 2-D numpy array
        :param margin: margin size defining the central area, default to the overlap size 34
        :return: the normalized diffusionprint data, 2-D numpy array with the same size of the input diffusionprint data
        '''
        v_min = np.min(diffusionprint[margin:-margin, margin:-margin])
        v_max = np.max(diffusionprint[margin:-margin, margin:-margin])
        return ((diffusionprint - v_min) / (v_max - v_min)).clip(0, 1)

    ''' -------------------- utilities -------------------- '''

    def load_weights(self, path, strict = True, device = 'cpu'):
        self.load_state_dict(torch.load(path, map_location=device), strict=strict)
        self.eval()
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
