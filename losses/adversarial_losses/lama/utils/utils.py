import torch


def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(),
                                        inputs=real_batch,
                                        create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0],
                                       -1).norm(2, dim=1)**2).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty
