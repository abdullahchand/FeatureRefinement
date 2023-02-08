def initialize_loss(loss, **kwargs):
    if loss == "adverserial_loss":
        from losses import adversarial_losses
        loss = adversarial_losses.get_adversarial_loss(**kwargs)
    elif loss == "l1_loss":
        from losses import distance_based_losses
        loss = distance_based_losses.get_distance_loss(**kwargs)
    elif loss == "perceptual_loss":
        from losses import perceptual_losses
        loss = perceptual_losses.get_perceptual_loss(**kwargs)
    elif loss == "ffl":
        from losses import focal_freq_loss
        loss = focal_freq_loss.get_freq_loss(**kwargs)
    else:
        raise ValueError(f'Unexpected transform loss {loss}')

    return loss
