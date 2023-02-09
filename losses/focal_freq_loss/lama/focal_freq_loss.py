import torch.nn as nn
import torch.nn.functional as F
from focal_frequency_loss import FocalFrequencyLoss as focalfreqloss
from models.utils.mask_utils import get_reigon_coordinates_for_cropping
import torch
import cv2
import os
# from torchvision.ops import masks_to_boxes
# import matplotlib.pyplot as plt
# import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


class FFL(nn.Module):

    def __init__(self,
                 alpha=1.0,
                 loss_weight=1.0,
                 ave_spectrum=False,
                 log_matrix=False,
                 batch_matrix=False,
                 patch_factor=1,
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.patch_factor = patch_factor
        self.ffl = focalfreqloss(loss_weight=2.0, alpha=1.0)

    def forward(self, image, predicted_image, mask, **kwargs):
        m = mask.expand(-1, 3, -1, -1)
        
        i = image.clone()
        p = predicted_image.clone()

        i[m==0] = 0
        p[m==0] = 0

        ffl = self.ffl(i,p)

        return {
            "loss": {
                "ffl": ffl
            },
            "values_to_log": dict({"FFl":ffl})
        }