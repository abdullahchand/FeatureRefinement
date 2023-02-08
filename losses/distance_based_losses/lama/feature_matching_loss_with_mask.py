from typing import List
from unittest import result

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaksedFeatureMatchingLoss(nn.Module):

    def __init__(self, weight=1, use_mask = False, **kwargs):
        super(MaksedFeatureMatchingLoss, self).__init__()
        self.weight = weight
        self.use_mask = use_mask

    def forward(self,
                discr_fake_features: List[torch.Tensor],
                discr_real_features: List[torch.Tensor],
                mask=None,
                **kwargs):
        if not self.use_mask or mask is None:
            res = torch.stack([
                F.mse_loss(fake_feat,
                           target_feat) for fake_feat, target_feat in zip(
                               discr_fake_features, discr_real_features)
            ]).mean()
        else:
            res = 0
            norm = 0
            for fake_feat, target_feat in zip(discr_fake_features,
                                              discr_real_features):
                cur_mask = F.interpolate(mask,
                                         size=fake_feat.shape[-2:],
                                         mode='bilinear',
                                         align_corners=False)
                error_weights = 1 - cur_mask
                cur_val = ((fake_feat - target_feat).pow(2) *
                           error_weights).mean()
                res = res + cur_val
                norm += 1
            res = res / norm
        res = res * self.weight
        return {
            "loss": {
                "masked_feature_matching_loss": res
            },
            "values_to_log": dict()
        }
