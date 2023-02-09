import numpy as np
import cv2
from utils.jpg_to_exr import return_jpg_to_exr

def convert_image(image):
    img = image
    out_img = img.copy().astype('float32') / 255
    return img, out_img


def tensor_to_numpy(image,pretrained_lama = True):
        
    inter_output = image.numpy()
    inter_output = inter_output.squeeze(0)
    inter_output = np.moveaxis(inter_output,0,-1)
    if pretrained_lama:
        # reverting preprocessing steps
        inter_output = np.clip(inter_output * 255, 0, 255).astype('uint8')
        inter_output = return_jpg_to_exr(inter_output)
    return inter_output

def save_image(image,save_path,pretrained_lama=True):
    inter_output = tensor_to_numpy(image,pretrained_lama)
    cv2.imwrite(save_path, inter_output)