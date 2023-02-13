import torch
import sys
from .training.modules.ffc import FFCResNetGenerator


def load_model_old_lama(model_param, checkpoint_path,device):
    generator = FFCResNetGenerator(**model_param)

    # state = torch.load(checkpoint_path, map_location="cuda")
    # generator.load_state_dict(state['state_dict'], strict=True)
    # generator.on_load_checkpoint(state)

    checkpoint = torch.load(checkpoint_path,map_location = device)
    # del checkpoint["optimizer_states"]

    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        if k.startswith("discriminator."):
            del state_dict[k]

    for k in list(state_dict.keys()):
        if k.startswith("loss_"):
            del state_dict[k]

    new_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith("generator."):
            new_state_dict[k.replace("generator.", "")] = state_dict[k]

    generator.load_state_dict(new_state_dict)
    generator.eval()

    return generator

def load_model_quixel_lama(model_param, checkpoint_path,device):
    
    generator = FFCResNetGenerator(**model_param)
    # generator = torch.nn.DataParallel(generator)
    state_dict = torch.load(checkpoint_path,map_location = device)
    for k in list(state_dict['generator'].keys()):
        if k.startswith("module."):
            state_dict['generator'][k.replace("module.", "")] = state_dict['generator'][k]
            del state_dict['generator'][k]  # remove original key-value pair
    # print(state_dict['generator'].keys())
    generator.load_state_dict(state_dict['generator'])
    # print(generator)
    generator.eval()
    return generator




def load_albedo_model(checkpoint_path,old_lama=False,device = torch.device("cpu")):
    model_param = {
        "input_nc": 4,
        "output_nc": 3,
        "ngf": 64,
        "n_downsampling": 3,
        "n_blocks": 18,
        "add_out_act": "sigmoid",
        "init_conv_kwargs": {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        "downsample_conv_kwargs": {
            "ratio_gin": 0,
            "ratio_gout": 0,
            "enable_lfu": False,
        },
        "resnet_conv_kwargs": {
            "ratio_gin": 0.75,
            "ratio_gout": 0.75,
            "enable_lfu": False,
        },
    }

    if old_lama:
        return load_model_old_lama(model_param, checkpoint_path,device = device)
    else:
        return load_model_quixel_lama(model_param, checkpoint_path,device = device)



def load_allmaps_model(checkpoint_path, old_lama=False,device = torch.device("cpu") ):
    
    model_param = {
        "input_nc": 9,
        "output_nc": 8,
        "ngf": 64,
        "n_downsampling": 3,
        "n_blocks": 18,
        "add_out_act": "sigmoid",
        "init_conv_kwargs": {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        "downsample_conv_kwargs": {
            "ratio_gin": 0,
            "ratio_gout": 0,
            "enable_lfu": False,
        },
        "resnet_conv_kwargs": {
            "ratio_gin": 0.75,
            "ratio_gout": 0.75,
            "enable_lfu": False,
        },
    }

    if old_lama:
        return load_model_old_lama(model_param, checkpoint_path,device = device)
    else:
        return load_model_quixel_lama(model_param, checkpoint_path,device = device)



if __name__ == "__main__":
    
    trained_checkpoint_path = "./pretrained_models/old_lama_all_maps.ckpt"
    pretrained_lama = "./pretrained_models/pretrained_original_lama_albedo_only.ckpt"

    pretrained_old_lama = load_albedo_model(pretrained_lama, old_lama=True)
    pretrained_old_lama_all = load_allmaps_model(trained_checkpoint_path, old_lama=True)

    checkpoint_path_quixel_lama_albedo_only = "./pretrained_models/quixel_lama_albedo_only_19999.pth"
    checkpoint_path_quixel_all_maps = "./pretrained_models/quixel_lama_all_maps_44999.pth"

    quixel_lama_albedo_only = load_albedo_model(checkpoint_path_quixel_lama_albedo_only)
    quixel_lama_all_maps = load_allmaps_model(checkpoint_path_quixel_all_maps)


    



