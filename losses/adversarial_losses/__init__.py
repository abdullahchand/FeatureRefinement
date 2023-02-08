def get_adversarial_loss(kind, **kwargs):
    if kind == 'non_saturating_with_r1':
        from losses.adversarial_losses.lama.non_saturating_with_r1 import NonSaturatingWithR1
        return NonSaturatingWithR1(**kwargs)
    elif kind == 'adversarial_bce_loss':
        from losses.adversarial_losses.lama.adversarial_bce_loss import BCELoss
        return BCELoss(**kwargs)

    return None
