import os 
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from torch.optim import Adam, SGD 
import json
import cv2
import numpy as np

import torch
from .models.model_loading import load_albedo_model
from .models.training.modules.ffc import FFCResnetBlock
from .utils.resize_utils import resize_mask, downscale, resize_image
from .utils.mask_utils import convert_to_1D
from torchvision import transforms
from .losses import initialize_loss
from .utils.refiner import _infer
from .utils.exr_to_jpg import return_exr_to_jpg
from .utils.jpg_to_exr import return_jpg_to_exr
from .utils.utils import convert_image, save_image,tensor_to_numpy
import os
from tqdm import tqdm

class FeatureRefinement:
    '''
    Feature refinement class used for postprocessing of Albedos.

    Parameters
    ----------
    checkpoint_path -> path to the model checkpoint
    pretrained_lama -> wether to use pretrained lama.
    config_path -> Path to the config file containing losses configs , default -> "sample_config.json"
    use_cuda -> Wether to use gpu or not
    model -> If given, uses this model istead of loading a model from checkpoint path.
    save_intermediate_output -> Wether to save the intermediate results.
    save_path -> path to save intermediate results
    '''
    def __init__(
        self, 
        checkpoint_path,
        use_cuda,
        config_path="sample_config.json",
        model = None,
        pretrained_lama=False,
        save_intermediate_output = False,
        save_path = "results/"
    ) -> None:
        # configuring device
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        self.save_intermediate_output = save_intermediate_output
        self.save_path = save_path

        self.transform = transforms.Compose([
        transforms.ToTensor()
        ])

        # loading model configuration
        self.configs = json.load(open(config_path, 'r'))
        self.losses = {}
        for key,loss in enumerate(self.configs["losses"]):
            self.losses[self.configs["losses"][loss]["kind"]] = initialize_loss(loss = loss, **self.configs["losses"][loss])
        print("Found the following losses : " ,self.losses)
        self.checkpoint_path = checkpoint_path

        # loading model if model is already passed then use the model.
        if model is None:
            self.pretrained_lama = pretrained_lama
            self.model=load_albedo_model(checkpoint_path=self.checkpoint_path,old_lama = self.pretrained_lama,device=self.device)
        else:
            self.model = model
        
        self.model.to("cpu")
        # Load and split model
        self.model.eval()
        self.encoder, self.decoder = self.split_model(self.model)
        self.encoder = self.freeze_module(self.encoder)
        self.decoder = self.freeze_module(self.decoder)
        self.decoder.to(self.device)
        #self.encoder.to("cpu")
        self.encoder.to(self.device)

    def split_model(self,model):
        '''
        Helper function to split the lama model into encoder and decoder.

        Parameters
        ----------
        model -> Loaded lama model

        Returns
        -------
        encoder -> nn.Module
        decoder -> nn.Module
        '''
        n_resnet_blocks = 0
        first_resblock_ind = 0
        found_first_resblock = False
        for idl in range(len(model.model)):
            if isinstance(model.model[idl], FFCResnetBlock):
                n_resnet_blocks += 1
                found_first_resblock = True
            elif not found_first_resblock:
                first_resblock_ind += 1
        encoder = model.model[0:first_resblock_ind]
        decoder = model.model[first_resblock_ind:]
        return encoder,decoder
    
    def freeze_module(self,module):
        '''
        Helper function to freeze a model or part of model.

        Parameters
        ---------
        module -> The module to be frozen

        Returns
        ------
        module -> Frozen module
        '''

        for param in module.parameters():
            param.requires_grad = False
        return module
    
    def unfreeze_module(self,module):
        '''
        Helper function to unfreeze a model or part of model.

        Parameters
        ---------
        module -> The module to be unfrozen

        Returns
        ------
        module -> unFrozen module
        '''
        
        for param in module.parameters():
            param.requires_grad = True
        return module


    def create_images_pyramid(self,high_res_image, high_res_mask, smallest_size, n_steps, difference = 128):
        '''
        Helper function to create images and masks of different scales to be used in iterative  fixing of the high res image

        Parameters
        ---------
        high_res_image -> The actual high res image
        high_res_mask -> The actual high res mask
        smallest_size -> The size of the startin low reference image i.e the smallest size of image
        n_steps -> defines the desired number of images to be produced. If an image during a step becomes smaller than the smallest size, it returns the images and masks created before this point.
        difference -> difference between each step.  Ex. If image size is (1024,1024), n_step is 3, smalles_size is (256,256) and difference is 256, It will produce image sizes of (1024,768,512)

        Returns
        -------
        images -> The images created -> []
        masks -> The masks created -> []
        scales -> The scales of respective mages and masks -> []
        '''
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



    def feature_refinement(self, high_res_image, mask, size = (512,512),n_iterations = 1, n_steps = 0,difference = 256, lr = 0.002,**kwargs):
        '''
        Performs the feature refinement on the given image and mask.
        It starts by creating images defined in the class through n_steps, difference and

        Parameters
        ----------
        high_res_image -> The actual high res map
        mask -> The actual high res mask
        size -> starting size / size of the first reference image to be inpainted.
        n_steps -> If we require step based refinement.
        difference -> The difference between each image step. Ex. If image size is (1024,1024), n_step is 3, smalles_size is (256,256) and difference is 256, It will produce image sizes of (1024,768,512)
        n_iterations -> number of iterations to perform for optimisation
        lr -> Learning rate of the model.

        Returns
        -------
        Returns the High res inpainted image -> nnumpy array
        '''
        # Conver tot jpg if pretrained lama
        if self.pretrained_lama:
            high_res_image = return_exr_to_jpg(high_res_image)
            _, high_res_image = convert_image(high_res_image)

        # Get images of different scales
        images,masks,scales = self.create_images_pyramid(high_res_image,mask,smallest_size=size,n_steps=n_steps,difference=difference)

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
        low_res_output = low_res_resized_mask * low_res_output + (1 - low_res_resized_mask) * low_res_image
        low_res_output = low_res_output.cpu().detach()

        if self.save_intermediate_output:
            curr_save_path = os.path.join(self.save_path,"low_res-"+str(size)+".exr")
            save_image(low_res_output,curr_save_path,pretrained_lama=self.pretrained_lama)

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
            #self.debug(high_res_image, resized_mask)
            # Get highres output
            high_res_prediction,without_refinement = _infer(high_res_image,resized_mask,self.encoder,self.decoder,current_reference,self.device,id,n_iters=n_iterations,downsize=current_scale,lr = lr, losses=self.losses)
            high_res_prediction = high_res_prediction.cpu().detach()
            
            # Set current reference as high res refence.
            current_reference = high_res_prediction
            current_scale = scale

            if self.save_intermediate_output:
                curr_save_path = os.path.join(self.save_path,str(id)+"-refined-"+str(scale)+".exr")
                save_image(current_reference,curr_save_path,pretrained_lama=self.pretrained_lama)

                curr_save_path = os.path.join(self.save_path,str(id)+"-unrefined-"+str(scale)+".exr")
                save_image(without_refinement.cpu().detach(),curr_save_path,pretrained_lama=self.pretrained_lama)


        current_reference = tensor_to_numpy(current_reference,pretrained_lama=self.pretrained_lama)
        
        return current_reference
    
    def debug(self, image, mask):
        masked_image = image * (1 - mask)
        masked_image = torch.cat([masked_image, mask], dim=1)
        masked_image = masked_image.to("cpu")

        a=1
