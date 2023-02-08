import torch.nn as nn
import torch.nn.functional as F


class MaskedL2Loss(nn.Module):

    def __init__(self,
                 weight_known,
                 weight_missing,
                 reduction='none',
                 **kwargs):
        super().__init__()
        self.weight_known = weight_known
        self.weight_missing = weight_missing
        self.reduction = reduction

    def forward(self, image, predicted_image, mask, **kwargs):
        per_pixel_l2 = F.mse_loss(predicted_image,
                                  image,
                                  reduction=self.reduction)
        pixel_weights = mask * self.weight_missing + (1 -
                                                      mask) * self.weight_known
        return {
            "loss": {
                "masked_l2_loss": (pixel_weights * per_pixel_l2).mean()
            },
            "values_to_log": dict()
        }