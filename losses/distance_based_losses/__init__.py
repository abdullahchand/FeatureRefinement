def get_distance_loss(kind, **kwargs):
    if kind == 'masked_feature_matching_loss':
        from losses.distance_based_losses.lama.feature_matching_loss_with_mask import MaksedFeatureMatchingLoss
        return MaksedFeatureMatchingLoss(**kwargs)
    elif kind == 'masked_l1_loss':
        from losses.distance_based_losses.lama.masked_l1_loss import MaskedL1Loss
        return MaskedL1Loss(**kwargs)
    elif kind == 'masked_l2_loss':
        from losses.distance_based_losses.lama.masked_l2_loss import MaskedL2Loss
        return MaskedL2Loss(**kwargs)

    return None
