from feature_refinement import FeatureRefinement
import cv2
from torchvision import transforms
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

FR = FeatureRefinement(config_path="sample_config.json",checkpoint_path='/home/abdul/Epic/Projects/FeatureRefinement/data/pretrained_model/pretrained_original_lama_albedo_only.ckpt',use_cuda=True,pretrained_lama=True)

def load_maps(path):
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    return image
transform = transforms.Compose([
        transforms.ToTensor()
        ])
mask = cv2.imread('data/1024mask.png')
image = load_maps("data/1024.exr")
FR.featur_refinement(image,mask,size=(512,512))