def get_freq_loss(kind, **kwargs):

    if kind == 'focal_frequency_loss':
        from losses.focal_freq_loss.lama.focal_freq_loss import FFL
        return FFL(**kwargs)

    return None