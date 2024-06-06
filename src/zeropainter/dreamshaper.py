import src.smplfusion
from src.smplfusion import scheduler
from src.smplfusion.common import *
from types import SimpleNamespace

import importlib
from omegaconf import OmegaConf
import torch
import safetensors
import safetensors.torch

from os.path import dirname

print("Loading model: Dreamshaper Inpainting V8")
PROJECT_DIR = dirname(dirname(dirname(dirname(__file__))))
print(PROJECT_DIR)
CONFIG_FOLDER = f"{PROJECT_DIR}/lib/smplfusion/config/"
MODEL_FOLDER = f"{PROJECT_DIR}/assets/models/"
ASSETS_FOLDER = f"{PROJECT_DIR}/assets/"


def get_inpainting_condition(model, image, mask):
    latent_size = [x // 8 for x in image.size]
    condition_x0 = (
        model.vae.encode(image.torch().cuda() * ~mask.torch(0).bool().cuda()).mean
        * model.config.scale_factor
    )
    condition_mask = mask.resize(latent_size[::-1]).cuda().torch(0).bool().float()
    return torch.cat([condition_mask, condition_x0], 1)


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def load_obj(path):
    objyaml = OmegaConf.load(path)
    return get_obj_from_str(objyaml["__class__"])(**objyaml.get("__init__", {}))


def get_t2i_model():
    model_t2i = SimpleNamespace()
    state_dict = safetensors.torch.load_file(
        f"{MODEL_FOLDER}/dreamshaper/dreamshaper_8.safetensors", device="cuda"
    )

    model_t2i.config = OmegaConf.load(f"{CONFIG_FOLDER}/ddpm/v1.yaml")
    model_t2i.unet = load_obj(f"{CONFIG_FOLDER}/unet/v1.yaml").eval().cuda()
    model_t2i.vae = load_obj(f"{CONFIG_FOLDER}/vae.yaml").eval().cuda()
    model_t2i.encoder = load_obj(f"{CONFIG_FOLDER}/encoders/clip.yaml").eval().cuda()

    extract = lambda state_dict, model: {
        x[len(model) + 1 :]: y for x, y in state_dict.items() if model in x
    }
    unet_state = extract(state_dict, "model.diffusion_model")
    encoder_state = extract(state_dict, "cond_stage_model")
    vae_state = extract(state_dict, "first_stage_model")

    model_t2i.unet.load_state_dict(unet_state, strict=False)
    model_t2i.encoder.load_state_dict(encoder_state, strict=False)
    model_t2i.vae.load_state_dict(vae_state, strict=False)
    model_t2i.unet = model_t2i.unet.requires_grad_(False)
    model_t2i.encoder = model_t2i.encoder.requires_grad_(False)
    model_t2i.vae = model_t2i.vae.requires_grad_(False)

    model_t2i.schedule = scheduler.linear(
        model_t2i.config.timesteps,
        model_t2i.config.linear_start,
        model_t2i.config.linear_end,
    )
    return model_t2i, unet_state


def get_inpainting_model():
    model_inp = SimpleNamespace()
    state_dict = safetensors.torch.load_file(
        f"{MODEL_FOLDER}/dreamshaper/dreamshaper_8Inpainting.safetensors", device="cuda"
    )

    model_inp.config = OmegaConf.load(f"{CONFIG_FOLDER}/ddpm/v1.yaml")
    model_inp.unet = load_obj(f"{CONFIG_FOLDER}/unet/inpainting/v1.yaml").eval().cuda()
    model_inp.vae = load_obj(f"{CONFIG_FOLDER}/vae.yaml").eval().cuda()
    model_inp.encoder = load_obj(f"{CONFIG_FOLDER}/encoders/clip.yaml").eval().cuda()

    extract = lambda state_dict, model: {
        x[len(model) + 1 :]: y for x, y in state_dict.items() if model in x
    }
    unet_state = extract(state_dict, "model.diffusion_model")
    encoder_state = extract(state_dict, "cond_stage_model")
    vae_state = extract(state_dict, "first_stage_model")

    model_inp.unet.load_state_dict(unet_state, strict=False)
    model_inp.encoder.load_state_dict(encoder_state, strict=False)
    model_inp.vae.load_state_dict(vae_state, strict=False)
    model_inp.unet = model_inp.unet.requires_grad_(False)
    model_inp.encoder = model_inp.encoder.requires_grad_(False)
    model_inp.vae = model_inp.vae.requires_grad_(False)

    model_inp.schedule = scheduler.linear(
        model_inp.config.timesteps,
        model_inp.config.linear_start,
        model_inp.config.linear_end,
    )
    return model_inp, unet_state
