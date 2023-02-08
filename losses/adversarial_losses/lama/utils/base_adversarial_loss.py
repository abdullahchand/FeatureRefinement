from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAdversarialLoss(nn.Module):

    def pre_generator_step(
            self,
            image: torch.Tensor,
            predicted_image: torch.Tensor,
            generator: nn.Module,
            discriminator: nn.Module,
            **kwargs
    ):
        """
        Prepare for generator step
        :param image: Tensor, a batch of real samples
        :param predicted_image: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def pre_discriminator_step(
            self,
            image: torch.Tensor,
            predicted_image: torch.Tensor,
            generator: nn.Module,
            discriminator: nn.Module,
            **kwargs
    ):
        """
        Prepare for discriminator step
        :param image: Tensor, a batch of real samples
        :param predicted_image: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def generator_loss(
            self,
            image: torch.Tensor,
            predicted_image: torch.Tensor,
            discr_real_pred: torch.Tensor,
            discr_fake_pred: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate generator loss
        :param image: Tensor, a batch of real samples
        :param predicted_image: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for image
        :param discr_fake_pred: Tensor, discriminator output for predicted_image
        :param mask: Tensor, actual mask, which was at input of generator when making predicted_image
        :return: total generator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def discriminator_loss(
            self,
            image: torch.Tensor,
            predicted_image: torch.Tensor,
            discr_real_pred: torch.Tensor,
            discr_fake_pred: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate discriminator loss and call .backward() on it
        :param image: Tensor, a batch of real samples
        :param predicted_image: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for image
        :param discr_fake_pred: Tensor, discriminator output for predicted_image
        :param mask: Tensor, actual mask, which was at input of generator when making predicted_image
        :return: total discriminator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def interpolate_mask(self, mask, shape):
        assert mask is not None
        assert self.allow_scale_mask or shape == mask.shape[-2:]
        if shape != mask.shape[-2:] and self.allow_scale_mask:
            if self.mask_scale_mode == "maxpool":
                mask = F.adaptive_max_pool2d(mask, shape)
            else:
                mask = F.interpolate(mask,
                                     size=shape,
                                     mode=self.mask_scale_mode)
        return mask
