import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import numpy

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class ResNetPL(nn.Module):

    def __init__(self,
                 weight=1,
                 use_cuda=False,
                 weights_path=None,
                 kind='resnet50dilated',
                 normalise = True,
                 displacement_with_contrast = False,
                 device_id=-1):
        super().__init__()
        self.impl = models.load_encoder(kind=kind, weights_path=weights_path)
        self.device_id = device_id
        if use_cuda:
            if device_id >= 0:
                self.impl.to("cuda:" + str(device_id))
            else:
                self.impl.cuda()
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight
        self.displacement_with_contrast = displacement_with_contrast

    def forward(self, predicted_image, image, original_displacement_with_contrast, \
                    predicted_displacement_with_contrast, experiment_configs, **kwargs):
        input_maps_config = experiment_configs['input_maps_config']
        
        if self.displacement_with_contrast:
            predicted_image = torch.cat((predicted_image,predicted_displacement_with_contrast),dim = 1)
            image = torch.cat((image,original_displacement_with_contrast),dim = 1)
            input_maps_config = input_maps_config + experiment_configs['additional_maps']

        
        current_channel = 0
        total_loss = None
        for map in input_maps_config:
            map_name = list(map.keys())[0]
            channels_in_map = map[map_name]
            to_channel = current_channel + channels_in_map
            pred = predicted_image[:, current_channel:to_channel, :, :]
            target = image[:, current_channel:to_channel, :, :]

            if pred.shape[1] == 1:
                    pred = pred.repeat(1, 3, 1, 1)
                    target = target.repeat(1, 3, 1, 1)
            if experiment_configs['normalise']:
                pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
                target = (target -
                        IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)
            input_device_id = pred.get_device()
            if input_device_id != self.device_id and self.device_id != -1:
                pred = pred.to('cuda:' + str(self.device_id))
                target = target.to('cuda:' + str(self.device_id))
            pred_feats = self.impl(pred, return_feature_maps=True)
            target_feats = self.impl(target, return_feature_maps=True)

            result = torch.stack([
                F.mse_loss(cur_pred, cur_target)
                for cur_pred, cur_target in zip(pred_feats, target_feats)
            ]).sum() * self.weight

            if input_device_id != self.device_id and self.device_id != -1:
                pred = pred.to('cuda:' + str(input_device_id))
                target = target.to('cuda:' + str(input_device_id))
                result = result.to('cuda:' + str(input_device_id))

            if total_loss is None:
                total_loss = result
            else:
                total_loss += result
            current_channel = to_channel

        return {
            "loss": {
                "resnet_pl": total_loss #/ len(input_maps_config)
            },
            "values_to_log": dict()
        }
