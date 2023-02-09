# POST  PROCESSING USING FEATURE REFINEMENT

## Description

This repository uses feature refinement on lama model to improve the output of the results. It uses the fact that lama is trained on a smaller patch size and inpaints better on that patch size. To inpaint lager patch size we add a loss between inpainting on smaller patch size and the inpainted larger patch, resized to the smaller patch size. For more information please refer : https://arxiv.org/abs/2206.13644

## Installing dependency

Please run the 'lama' envoirment. To create the lama envoirment :-
```
conda env create -f conda_env.yml

conda activate lama

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
```

## Run

Create a config file, similar to sample_config.json, defining the losses and thier configurations


```json

{
    "losses": {
        "l1_loss": {
            "kind" : "masked_l1_loss",
            "weight_missing": 0,
            "weight_known": 10,
            "displacement_with_contrast": true
        }
    }
}

```

you can add more losses by defining the following structure :-

```json
{
    "losses": {
        "loss_type": {
            # loss configurations
        }
    }
}

```

Currently we support the following losses : -

[adverserial_loss , l1_loss, perceptual_loss, ffl]

You can refer to the lama training code for each loss configuration.
(Please ensure that each loss has 'kind' keyword definin the type of the loss.)


If we dont want to specify losses and use the default loss we can define the sample_config.json as :-

```json
{
    "losses": {
    }
}
```

To run and use in code:-

```python
from feature_refinement import FeatureRefinement
import cv2
from torchvision import transforms
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

FR = FeatureRefinement(config_path="sample_config.json",checkpoint_path='/home/abdul/Epic/Projects/FeatureRefinement/data/pretrained_model/pretrained_original_lama_albedo_only.ckpt',use_cuda=True,pretrained_lama=True,save_intermediate_output=True,save_path="results/"
,n_steps=3,difference=256)

def load_maps(path):
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    return image
transform = transforms.Compose([
        transforms.ToTensor()
        ])
mask = cv2.imread('data/1024mask.png') # High res mask
image = load_maps("data/1024.exr") # High res image
FR.feature_refinement(image,mask,size=(256,256),n_iterations=10)
```