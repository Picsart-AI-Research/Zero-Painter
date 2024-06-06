print ("Loading model: Dreamshaper Inpainting V8")

from os.path import dirname
import importlib
from omegaconf import OmegaConf
import torch
import safetensors
import safetensors.torch
import open_clip

PROJECT_DIR = dirname(dirname(dirname(dirname(__file__))))
LIB_DIR = dirname(dirname(__file__))
print (PROJECT_DIR)

CONFIG_FOLDER =  f'{LIB_DIR}/config/'
ASSETS_FOLDER =  f'{PROJECT_DIR}/assets/'
MODEL_FOLDER =  f'{ASSETS_FOLDER}/models/'

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(module, package=None), cls)
    except:
        return getattr(importlib.import_module('lib.' + module, package=None), cls)
def load_obj(path):
    objyaml = OmegaConf.load(path)
    return get_obj_from_str(objyaml['__class__'])(**objyaml.get("__init__", {}))
    
state_dict = safetensors.torch.load_file(f'{MODEL_FOLDER}/dreamshaper/dreamshaper_8Inpainting.safetensors')

config = OmegaConf.load(f'{CONFIG_FOLDER}/ddpm/v1.yaml')
unet = load_obj(f'{CONFIG_FOLDER}/unet/inpainting/v1.yaml').eval().cuda()
vae = load_obj(f'{CONFIG_FOLDER}/vae.yaml').eval().cuda()
encoder = load_obj(f'{CONFIG_FOLDER}/encoders/clip.yaml').eval().cuda()

extract = lambda state_dict, model: {x[len(model)+1:]:y for x,y in state_dict.items() if model in x}
unet_state = extract(state_dict, 'model.diffusion_model')
encoder_state = extract(state_dict, 'cond_stage_model')
vae_state = extract(state_dict, 'first_stage_model')

unet.load_state_dict(unet_state);
encoder.load_state_dict(encoder_state);
vae.load_state_dict(vae_state);

unet = unet.requires_grad_(False)
encoder = encoder.requires_grad_(False)
vae = vae.requires_grad_(False)