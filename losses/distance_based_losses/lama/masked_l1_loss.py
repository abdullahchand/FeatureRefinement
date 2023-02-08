import torch.nn as nn
import torch.nn.functional as F
import numpy
import torch

class MaskedL1Loss(nn.Module):

    def __init__(self,
                 weight_known,
                 weight_missing,
                 reduction='none',
                 displacement_with_contrast = False,
                 **kwargs):
        super(MaskedL1Loss, self).__init__()
        self.weight_known = weight_known
        self.weight_missing = weight_missing
        self.reduction = reduction
        self.displacement_with_contrast = displacement_with_contrast
        
    
    def forward(self, image, predicted_image, mask):
        
        per_pixel_l1 = F.l1_loss(predicted_image,
                                 image,
                                 reduction=self.reduction)
        pixel_weights = mask * self.weight_missing + (1 -
                                                      mask) * self.weight_known
        return (pixel_weights * per_pixel_l1).mean()