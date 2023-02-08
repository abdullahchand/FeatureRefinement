import numpy as np
from PIL import Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def GetNumberOfChannels(source):
    if len(source.shape) < 3:
        return 1
    return source.shape[2]

def Convert32BitTo8Bit(image):
    noOfChannels = GetNumberOfChannels(image)
    nimg = np.uint8(image)
    if noOfChannels == 1:  # If image is grayscale.
        nimg[:, :] = Bit32To8Operation(image[:, :])
    else:
        nimg[:, :, 0] = Bit32To8Operation(image[:, :, 0])
        nimg[:, :, 1] = Bit32To8Operation(image[:, :, 1])
        nimg[:, :, 2] = Bit32To8Operation(image[:, :, 2])
    if noOfChannels == 4:  # If image has an alpha channel
        nimg[:, :, 3] = Bit32To8Operation(image[:, :, 3])

    return nimg

def Bit32To8Operation(channel):
    channel[channel < 0.0] = 0.0
    channel[channel > 1.0] = 1.0
    ind = channel <= 0.0031308
    channel[ind] = (channel[ind]*12.92)*255.0
    ind = np.invert(ind)
    channel[ind] = (1.055*(channel[ind]**(1.0/2.4))-0.055) * 255.0
    return channel.astype('uint8')

def convert_exr_to_jpg(source, destination):
    stream = open(source, "rb")
    img = cv2.imdecode(np.asarray(
        bytearray(stream.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Convert32BitTo8Bit(img)
    if GetNumberOfChannels(img) == 4:
        imageData = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    im = Image.fromarray(img)
    im.save(destination, quality="web_maximum")



def return_exr_to_jpg(source):
    img = Convert32BitTo8Bit(source)
    if GetNumberOfChannels(img) == 4:
        imageData = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img