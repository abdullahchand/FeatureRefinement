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
from old_refinement import _infer
from utils.exr_to_jpg import return_exr_to_jpg
from utils.jpg_to_exr import return_jpg_to_exr
from utils.utils import convert_image

from tqdm import tqdm

class FeatureRefinement:
    def __init__(
        self,
        config_path, 
        checkpoint_path,
        use_cuda,
        model = None,
        pretrained_lama=False
    ) -> None:
        # configuring device
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
        transforms.ToTensor()
        ])

        # loading model configuration
        self.configs = json.load(open(config_path, 'r'))
        self.losses = []
        for key,loss in enumerate(self.configs["losses"]):
            self.losses.append(losses.initialize_loss(loss = loss, **self.configs["losses"][loss]))
        print(self.losses)
        self.checkpoint_path = checkpoint_path

        # loading model if model is already passed then use the model.
        if model is None:
            self.pretrained_lama = pretrained_lama
            self.model=load_albedo_model(checkpoint_path=self.checkpoint_path,old_lama = self.pretrained_lama,device=self.device)
        else:
            self.model = model
        self.model.eval()
        self.encoder, self.decoder = self.split_model(self.model)
        self.encoder = self.freeze_module(self.encoder)
        self.decoder = self.freeze_module(self.decoder)
        self.encoder.eval()
        self.decoder.eval()

    def _init_model(self):
        self.model = models.load_generator(**self.configs['generator'])

        # loading state dict
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)

        for k in list(state_dict['generator'].keys()):
            if k.startswith("module."):
                state_dict['generator'][k.replace("module.", "")] = state_dict['generator'][k]
                del state_dict['generator'][k]  # remove original key-value pair

        # loading model (generator weights)
        self.model.load_state_dict(state_dict['generator'])
        self.model = self.model.to(self.device)
        self.model.eval()       # swtich evaluation mode on

        # garbage collection
        del state_dict

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

    def featur_refinement(self, high_res_image, mask, size = (512,512),n_iterations = 1,lr = 0.002):
        if self.pretrained_lama:
            high_res_image = return_exr_to_jpg(high_res_image)
            _, high_res_image = convert_image(high_res_image)


        # Resize mask to fit the desired size
        low_res_resized_mask = resize_mask(mask, size=size)
        low_res_resized_mask = self.transform(low_res_resized_mask)
        low_res_resized_mask = low_res_resized_mask.unsqueeze(0)

        # Convert image to tensor and create masked image for input
        low_res_image = resize_image(high_res_image,size=size)
        low_res_image = self.transform(low_res_image)
        low_res_image = low_res_image.unsqueeze(0)
        masked_image = low_res_image * (1 - low_res_resized_mask)
        masked_image = torch.cat([masked_image, low_res_resized_mask], dim=1)

        # Get low res output
        low_res_output = self.model(masked_image)
        low_res_output = low_res_output.detach()
        result_output = low_res_output.numpy()
        result_output = result_output.squeeze(0)
        result_output = np.moveaxis(result_output,0,-1)
        if self.pretrained_lama:
            # reverting preprocessing steps
            result_output = np.clip(result_output * 255, 0, 255).astype('uint8')
            result_output = return_jpg_to_exr(result_output)

        # get high res embeddings
        mask = convert_to_1D(mask)
        resized_mask = self.transform(mask)
        resized_mask = resized_mask.unsqueeze(0)

        # Convert image to tensor and create masked image for input
        high_res_image = self.transform(high_res_image)
        high_res_image = high_res_image.unsqueeze(0)
        masked_image = high_res_image * (1 - resized_mask)
        masked_image = torch.cat([masked_image, resized_mask], dim=1)
        high_res_prediction,without_refinement = _infer(high_res_image,resized_mask,self.encoder,self.decoder,low_res_output,high_res_image.shape,self.device,1,n_iters=n_iterations,downsize=size,lr = lr)

        inpainted_high_res = inpainted_high_res.cpu().detach().numpy()
        inpainted_high_res = inpainted_high_res.squeeze(0)
        inpainted_high_res = np.moveaxis(inpainted_high_res,0,-1)


        high_result_output = high_res_prediction.cpu().detach().numpy()
        high_result_output = high_result_output.squeeze(0)
        high_result_output = np.moveaxis(high_result_output,0,-1)
        if self.pretrained_lama:
            # reverting preprocessing steps
            high_result_output = np.clip(high_result_output * 255, 0, 255).astype('uint8')
            high_result_output = return_jpg_to_exr(high_result_output)

            inpainted_high_res = np.clip(inpainted_high_res * 255, 0, 255).astype('uint8')
            inpainted_high_res = return_jpg_to_exr(inpainted_high_res)
        cv2.imwrite("HighResOutput-jpg.exr", high_result_output)
        cv2.imwrite("HighResOutput-without-jpg.exr", inpainted_high_res)
        return high_res_prediction