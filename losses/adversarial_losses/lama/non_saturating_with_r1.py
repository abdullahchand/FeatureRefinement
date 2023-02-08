from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.adversarial_losses.lama.utils.utils import make_r1_gp
from losses.adversarial_losses.lama.utils.base_adversarial_loss import BaseAdversarialLoss


class NonSaturatingWithR1(BaseAdversarialLoss):

    def __init__(self,
                 gp_coef=5,
                 weight=1,
                 mask_as_fake_target=False,
                 allow_scale_mask=False,
                 mask_scale_mode='nearest',
                 extra_mask_weight_for_gen=0,
                 use_unmasked_for_gen=True,
                 use_unmasked_for_discr=True,
                 **kwargs):
        super().__init__()
        self.gp_coef = gp_coef
        self.weight = weight
        # use for discr => use for gen;
        # otherwise we teach only the discr to pay attention to very small difference
        assert use_unmasked_for_gen or (not use_unmasked_for_discr)
        # mask as target => use unmasked for discr:
        # if we don't care about unmasked regions at all
        # then it doesn't matter if the value of mask_as_fake_target is true or false
        assert use_unmasked_for_discr or (not mask_as_fake_target)
        self.use_unmasked_for_gen = use_unmasked_for_gen
        self.use_unmasked_for_discr = use_unmasked_for_discr
        self.mask_as_fake_target = mask_as_fake_target
        self.allow_scale_mask = allow_scale_mask
        self.mask_scale_mode = mask_scale_mode
        self.extra_mask_weight_for_gen = extra_mask_weight_for_gen

    def generator_loss(self, image: torch.Tensor, predicted_image: torch.Tensor,
                       discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                       mask=None, **kwargs) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_loss = F.softplus(-discr_fake_pred)
        if (self.mask_as_fake_target and self.extra_mask_weight_for_gen > 0) or \
                not self.use_unmasked_for_gen:  # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            if not self.use_unmasked_for_gen:
                fake_loss = fake_loss * mask
            else:
                pixel_weights = 1 + mask * self.extra_mask_weight_for_gen
                fake_loss = fake_loss * pixel_weights
        return {
            "loss": {
                "non_saturating_with_r1_generator_loss":
                fake_loss.mean() * self.weight
            },
            "values_to_log": dict()
        }

    def pre_discriminator_step(self, image: torch.Tensor,
                               predicted_image: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module,
                               **kwargs):
        image.requires_grad = True

    def discriminator_loss(self, image: torch.Tensor, predicted_image: torch.Tensor,
                           discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                           mask=None, **kwargs) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        real_loss = F.softplus(-discr_real_pred)
        grad_penalty = make_r1_gp(discr_real_pred, image) * self.gp_coef
        fake_loss = F.softplus(discr_fake_pred)

        if not self.use_unmasked_for_discr or self.mask_as_fake_target:
            # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            # use_unmasked_for_discr=False only makes sense for fakes;
            # for reals there is no difference beetween two regions
            fake_loss = fake_loss * mask
            if self.mask_as_fake_target:
                fake_loss = fake_loss + (1 -
                                         mask) * F.softplus(-discr_fake_pred)

        sum_discr_loss = real_loss + grad_penalty + fake_loss
        metrics = dict(discr_real_gp=grad_penalty)
        return {
            "loss": {
                "non_saturating_with_r1_discriminator_loss":
                sum_discr_loss.mean()
            },
            "values_to_log": metrics
        }
