import torch
import torch.nn as nn
from torch.optim import Adam, SGD 
from kornia.filters import gaussian_blur2d
from kornia.geometry.transform import resize
from kornia.morphology import erosion
from torch.nn import functional as F
import numpy as np
import cv2
from tqdm import tqdm


def _pyrdown(im : torch.Tensor, downsize : tuple=None):
    """downscale the image"""
    if downsize is None:
        downsize = (im.shape[2]//2, im.shape[3]//2)
    assert im.shape[1] == 3, "Expected shape for the input to be (n,3,height,width)"
    # im = gaussian_blur2d(im, kernel_size=(5,5), sigma=(1.0,1.0))
    im = F.interpolate(im, size=downsize, mode='bilinear', align_corners=False)
    return im

def _pyrdown_mask(mask : torch.Tensor, downsize : tuple=None, eps : float=1e-8, blur_mask : bool=True, round_up : bool=True):
    """downscale the mask tensor
    Parameters
    ----------
    mask : torch.Tensor
        mask of size (B, 1, H, W)
    downsize : tuple, optional
        size to downscale to. If None, image is downscaled to half, by default None
    eps : float, optional
        threshold value for binarizing the mask, by default 1e-8
    blur_mask : bool, optional
        if True, apply gaussian filter before downscaling, by default True
    round_up : bool, optional
        if True, values above eps are marked 1, else, values below 1-eps are marked 0, by default True
    Returns
    -------
    torch.Tensor
        downscaled mask
    """

    if downsize is None:
        downsize = (mask.shape[2]//2, mask.shape[3]//2)
    assert mask.shape[1] == 1, "Expected shape for the input to be (n,1,height,width)"
    if blur_mask == True:
        mask = gaussian_blur2d(mask, kernel_size=(5,5), sigma=(1.0,1.0))
        mask = F.interpolate(mask, size=downsize,  mode='bilinear', align_corners=False)
    else:
        mask = F.interpolate(mask, size=downsize,  mode='bilinear', align_corners=False)
    if round_up:
        mask[mask>=eps] = 1
        mask[mask<eps] = 0
    else:
        mask[mask>=1.0-eps] = 1
        mask[mask<1.0-eps] = 0
    return mask

def _erode_mask(mask : torch.Tensor, ekernel : torch.Tensor=None, eps : float=1e-8):
    """erode the mask, and set gray pixels to 0"""
    if ekernel is not None:
        mask = erosion(mask, ekernel)
        mask[mask>=1.0-eps] = 1
        mask[mask<1.0-eps] = 0
    return mask


def _l1_loss(
    pred : torch.Tensor, pred_downscaled : torch.Tensor, ref : torch.Tensor, 
    mask : torch.Tensor, mask_downscaled : torch.Tensor, 
    image : torch.Tensor, on_pred : bool=True
    ):
    """l1 loss on src pixels, and downscaled predictions if on_pred=True"""
    loss = torch.mean(torch.abs(pred - image))
    if on_pred: 
        loss += torch.mean(torch.abs(pred_downscaled - ref))                
    return loss

def _infer(
    image : torch.Tensor, mask : torch.Tensor, 
    forward_front : nn.Module, forward_rears : nn.Module, 
    ref_lower_res : torch.Tensor, devices : list, 
    scale_ind : int, n_iters : int=15, lr : float=0.002,downsize = None, losses = None):

    """Performs models with refinement at a given scale.
    Parameters
    ----------
    image : torch.Tensor
        input image to be inpainted, of size (1,3,H,W)
    mask : torch.Tensor
        input inpainting mask, of size (1,1,H,W) 
    forward_front : nn.Module
        the front part of the inpainting network
    forward_rears : nn.Module
        the rear part of the inpainting network
    ref_lower_res : torch.Tensor
        the inpainting at previous scale, used as reference image
    devices : list
        list of available devices
    scale_ind : int
        the scale index
    n_iters : int, optional
        number of iterations of refinement, by default 15
    lr : float, optional
        learning rate, by default 0.002
    Returns
    -------
    torch.Tensor
        inpainted image
    """
    masked_image = image * (1 - mask)
    masked_image = torch.cat([masked_image, mask], dim=1)

    # mask = mask.repeat(1,3,1,1)
    if ref_lower_res is not None:
        ref_lower_res = ref_lower_res.detach()
    with torch.no_grad():
        z1,z2 = forward_front(masked_image)
    # models
    mask = mask.to(devices)
    ekernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)).astype(bool)).float()
    ekernel = ekernel.to(devices)
    image = image.to(devices)
    z1, z2 = z1.detach().to(devices), z2.detach().to(devices)
    z1.requires_grad, z2.requires_grad = True, True

    optimizer = Adam([z1,z2], lr=lr)
    # rescale the ref_low_res
    # rescaled_ref_low_res = resize(ref_lower_res, (image.shape[0], image.shape[1]))

    # input_feat = (z1,z2)
    count = 0
    without_refinement = None
    pbar = tqdm(range(n_iters), leave=False)
    for idi in pbar:
        optimizer.zero_grad()
        input_feat = (z1,z2)
        output_feat = forward_rears(input_feat)
        pred = output_feat
        if count == 0:
            without_refinement = output_feat
        if ref_lower_res is None:
            break
        curr_loss = {}
        ######################### multi-scale #############################
        # scaled loss with downsampler
        pred_downscaled = _pyrdown(pred,downsize=downsize)
        mask_downscaled = _pyrdown_mask(mask, blur_mask=False, round_up=False,downsize=downsize)
        mask_downscaled = _erode_mask(mask_downscaled, ekernel=ekernel)
        repeated_mask = mask.repeat(1,3,1,1)
        repeated_mask_downscaled = mask_downscaled.repeat(1,3,1,1)
        
        curr_loss["ms_l1"] = _l1_loss(pred.to(devices), pred_downscaled.to(devices), ref_lower_res.to(devices), repeated_mask.to(devices), repeated_mask_downscaled.to(devices), image.to(devices), on_pred=True)
        
        if losses is not None:
            for key,loss in enumerate(losses):
                curr_loss[loss] = losses[loss](pred_downscaled.to(devices), ref_lower_res.to(devices), repeated_mask_downscaled.to(devices))["loss"][loss]
                
        loss = sum(curr_loss.values())
        pbar.set_description("Refining scale {} using scale {} ...current loss: {:.4f}".format(scale_ind+1, scale_ind, loss.item()))
        if idi < n_iters - 1:
            loss.backward()
            optimizer.step()
            del pred_downscaled
            del loss
            del pred
        count+=1
    # "pred" is the prediction after Plug-n-Play module
    inpainted = mask * pred + (1 - mask) * image
    inpainted = inpainted
    return inpainted, without_refinement