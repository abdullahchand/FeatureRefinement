def get_perceptual_loss(kind, **kwargs):
    if kind == 'perceptual_loss':
        from losses.perceptual_losses.lama.vgg19_perceptual_loss import VGG19PerceptualLoss
        return VGG19PerceptualLoss(**kwargs)
    elif kind == 'resnet_pl':
        from losses.perceptual_losses.lama.ade20k_segmentation_resnet_pl import ResNetPL
        return ResNetPL(**kwargs)

    return None
