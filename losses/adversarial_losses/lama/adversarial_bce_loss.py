from typing import Tuple, Dict

import torch
import torch.nn as nn
from losses.adversarial_losses.lama.utils.base_adversarial_loss import BaseAdversarialLoss


class BCELoss(BaseAdversarialLoss):

    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def generator_loss(
            self, discr_fake_pred: torch.Tensor,
            **kwargs) -> Dict[str, torch.Tensor]:
        real_mask_gt = torch.zeros(discr_fake_pred.shape).to(
            discr_fake_pred.device)
        fake_loss = self.bce_loss(discr_fake_pred, real_mask_gt) * self.weight
        return {
            "loss": {
                "adversarial_bce_loss_generator_loss": fake_loss
            },
            "values_to_log": dict()
        }

    def pre_discriminator_step(self, image: torch.Tensor,
                               predicted_image: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module,
                               **kwargs):
        image.requires_grad = True

    def discriminator_loss(
            self, mask: torch.Tensor, discr_real_pred: torch.Tensor,
            discr_fake_pred: torch.Tensor,
            **kwargs) -> Dict[str, torch.Tensor]:

        real_mask_gt = torch.zeros(discr_real_pred.shape).to(
            discr_real_pred.device)
        sum_discr_loss = (self.bce_loss(discr_real_pred, real_mask_gt) +
                          self.bce_loss(discr_fake_pred, mask)) / 2
        metrics = dict(discr_real_gp=0)
        return {
            "loss": {
                "adversarial_bce_loss_discriminator_loss": sum_discr_loss
            },
            "values_to_log": metrics
        }
