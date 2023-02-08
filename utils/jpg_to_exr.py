import numpy as np
from PIL import Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def GetNumberOfChannels(source):
    if len(source.shape) < 3:
        return 1
    return source.shape[2]

def Convert8BitTo32Bit(image):  # Works
    noOfChannels = GetNumberOfChannels(image)
    nimg = image.copy()
    nimg = np.float32(image)

    if noOfChannels == 1:  # If image is grayscale.
        nimg[:, :] = (nimg[:, :] / 255.0).astype('float32')
    else:
        nimg[:, :, 0] = (nimg[:, :, 0] / 255.0).astype('float32')
        nimg[:, :, 1] = (nimg[:, :, 1] / 255.0).astype('float32')
        nimg[:, :, 2] = (nimg[:, :, 2] / 255.0).astype('float32')

    if noOfChannels == 4:  # If image has an alpha channel
        nimg[:, :, 3] = (nimg[:, :, 3] / 255.0).astype('float32')

    nimg = np.where(True, 1.0 * ((nimg/1.0) ** 2.2), 0)
    
    return nimg


def convert_jpg_to_exr(source, destination):
    stream = open(source, "rb")
    img = cv2.imdecode(np.asarray(
        bytearray(stream.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = Convert8BitTo32Bit(img)
    cv2.imwrite(destination, img)

def return_jpg_to_exr(source):
    img = source
    img = Convert8BitTo32Bit(img)
    return img