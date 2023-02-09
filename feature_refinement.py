import os 
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from torch.optim import Adam, SGD 
import json
import cv2
import numpy as np

import torch

import models
from models.model_loading import load_albedo_model
from models.training.modules.ffc import FFCResnetBlock
from utils.resize_utils import resize_mask, downscale, resize_image
from utils.mask_utils import convert_to_1D
from torchvision import transforms
import losses
from utils.refiner import _infer
from utils.exr_to_jpg import return_exr_to_jpg
from utils.jpg_to_exr import return_jpg_to_exr
from utils.utils import convert_image
import os
from tqdm import tqdm

class FeatureRefinement:
    def __init__(
        self,
        config_path, 
        checkpoint_path,
        use_cuda,
        model = None,
        pretrained_lama=False,
        save_intermediate_output = False,
        save_path = "results/",
        n_steps = 0,
        difference = 256,
        lr = 0.002
    ) -> None:
        # configuring device
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        self.save_intermediate_output = save_intermediate_output
        self.save_path = save_path

        self.n_steps = n_steps
        self.difference = difference
        self.lr = lr

        self.transform = transforms.Compose([
        transforms.ToTensor()
        ])

        # loading model configuration
        self.configs = json.load(open(config_path, 'r'))
        self.losses = []
        for key,loss in enumerate(self.configs["losses"]):
            self.losses.append(losses.initialize_loss(loss = loss, **self.configs["losses"][loss]))
        print("Found the following losses : " ,self.losses)
        self.checkpoint_path = checkpoint_path

        # loading model if model is already passed then use the model.
        if model is None:
            self.pretrained_lama = pretrained_lama
            self.model=load_albedo_model(checkpoint_path=self.checkpoint_path,old_lama = self.pretrained_lama,device=self.device)
        else:
            self.model = model
        
        # Load and split model
        self.model.eval()
        self.encoder, self.decoder = self.split_model(self.model)
        self.encoder = self.freeze_module(self.encoder)
        self.decoder = self.freeze_module(self.decoder)
        self.decoder.to(self.device)
        self.encoder.to(self.device)

    def split_model(self,model):
        n_resnet_blocks = 0
        first_resblock_ind = 0
        found_first_resblock = False
        for idl in range(len(model.model)):
            if isinstance(model.model[idl], FFCResnetBlock):
                n_resnet_blocks += 1
                found_first_resblock = True
            elif not found_first_resblock:
                first_resblock_ind += 1
        encoder = model.model[0:first_resblock_ind+1]
        decoder = model.model[first_resblock_ind:]
        return encoder,decoder
    
    def freeze_module(self,module):
        for param in module.parameters():
            param.requires_grad = False
        return module
    
    def unfreeze_module(self,module):
        for param in module.parameters():
            param.requires_grad = True
        return module


    def create_images_pyramid(self,high_res_image, high_res_mask, smallest_size, n_steps, difference = 128):
        images = []
        masks = []
        scales = []
        difference = int(difference)
        h,w,_ = high_res_image.shape

        images.append(high_res_image)
        masks.append(convert_to_1D(high_res_mask))
        scales.append((h,w))

        new_h = h
        new_w = w
        for step in range(0,n_steps):

            new_h = new_h - difference
            new_w = new_w - difference

            if new_h > smallest_size[0] and new_w > smallest_size[1]:
                scales.append((new_h,new_w))
                images.append(resize_image(high_res_image, (new_h,new_w)))
                masks.append(resize_mask(high_res_mask, (new_h,new_w)))
                
            else:
                break
        
        images = images[::-1]
        masks = masks[::-1]
        scales = scales[::-1]

        return images, masks, scales

    def tensor_to_numpy(self,image):
        inter_output = image.numpy()
        inter_output = inter_output.squeeze(0)
        inter_output = np.moveaxis(inter_output,0,-1)
        if self.pretrained_lama:
            # reverting preprocessing steps
            inter_output = np.clip(inter_output * 255, 0, 255).astype('uint8')
            inter_output = return_jpg_to_exr(inter_output)
        return inter_output

    def save_image(self,image,save_path):
        inter_output = self.tensor_to_numpy(image)
        cv2.imwrite(save_path, inter_output)


    def feature_refinement(self, high_res_image, mask, size = (512,512),n_iterations = 1):
        # Conver tot jpg if pretrained lama
        if self.pretrained_lama:
            high_res_image = return_exr_to_jpg(high_res_image)
            _, high_res_image = convert_image(high_res_image)

        # Get images of different scales
        images,masks,scales = self.create_images_pyramid(high_res_image,mask,smallest_size=size,n_steps=self.n_steps,difference=self.difference)

        # get first reference image

        # Resize mask to fit the desired size
        low_res_resized_mask = resize_mask(mask, size=size)
        low_res_resized_mask = self.transform(low_res_resized_mask)
        low_res_resized_mask = low_res_resized_mask.to(self.device)
        low_res_resized_mask = low_res_resized_mask.unsqueeze(0)

        # Convert image to tensor and create masked image for input
        low_res_image = resize_image(high_res_image,size=size)
        low_res_image = self.transform(low_res_image)
        low_res_image = low_res_image.to(self.device)
        low_res_image = low_res_image.unsqueeze(0)
        masked_image = low_res_image * (1 - low_res_resized_mask)
        masked_image = torch.cat([masked_image, low_res_resized_mask], dim=1)

        # Get low res output
        
        low_res_output = self.model(masked_image)
        low_res_output = low_res_output.cpu().detach()
        current_reference = low_res_output
        current_scale = size
        for id,(inter_image,inter_mask,scale) in enumerate(zip(images,masks,scales)):

            # Convert mask to tensor
            resized_mask = self.transform(inter_mask)
            resized_mask = resized_mask.to(self.device)
            resized_mask = resized_mask.unsqueeze(0)

            

            # Convert image to tensor
            high_res_image = self.transform(inter_image)
            high_res_image = high_res_image.to(self.device)
            high_res_image = high_res_image.unsqueeze(0)
            
            # Get highres output
            high_res_prediction,without_refinement = _infer(high_res_image,resized_mask,self.encoder,self.decoder,current_reference,self.device,id,n_iters=n_iterations,downsize=current_scale,lr = self.lr)
            high_res_prediction = high_res_prediction.cpu().detach()
            
            # Set current reference as high res refence.
            current_reference = high_res_prediction
            current_scale = scale

            if self.save_intermediate_output:
                curr_save_path = os.path.join(self.save_path,str(id)+"-refined-"+str(scale)+".exr")
                self.save_image(current_reference,curr_save_path)

                curr_save_path = os.path.join(self.save_path,str(id)+"-unrefined-"+str(scale)+".exr")
                self.save_image(without_refinement.cpu().detach(),curr_save_path)


        current_reference = self.tensor_to_numpy(current_reference)
        
        return current_reference