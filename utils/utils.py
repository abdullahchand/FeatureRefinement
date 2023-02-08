import numpy as np
import cv2
def convert_image(image):
    img = image
    out_img = img.copy().astype('float32') / 255
    return img, out_img