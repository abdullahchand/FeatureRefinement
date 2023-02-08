import cv2
from kornia.geometry.transform import resize

from kornia.filters import gaussian_blur2d
# from kornia.geometry.transform import resize
# from kornia.morphology import erosion
from torch.nn import functional as F
from utils.mask_utils import convert_to_1D

def resize_mask(mask,size=(512,512)):
    resized_mask = cv2.resize(mask,size)
    resized_mask = convert_to_1D(resized_mask)
    return resized_mask

def resize_image(image,size=(512,512)):
    resized_img = cv2.resize(image,size)
    return resized_img

def downscale(im, downsize):
    """downscale the image"""
    # im = resize(im,size)
    if downsize is None:
        downsize = (im.shape[2]//2, im.shape[3]//2)
    assert im.shape[1] == 3, "Expected shape for the input to be (n,3,height,width)"
    im = gaussian_blur2d(im, kernel_size=(5,5), sigma=(1.0,1.0))
    im = F.interpolate(im, size=downsize, mode='bilinear', align_corners=False)
    return im