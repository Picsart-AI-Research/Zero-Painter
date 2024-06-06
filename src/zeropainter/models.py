from types import SimpleNamespace
from . import convert_diffusers
import safetensors.torch

import torch
from src.smplfusion import scheduler
from os.path import dirname

from omegaconf import OmegaConf
import importlib

def load_obj(objyaml):
    if "__init__" in objyaml:
        return get_obj_from_str(objyaml["__class__"])(**objyaml["__init__"])
    else:
        return get_obj_from_str(objyaml["__class__"])()


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(module, package=None), cls)
    except Exception as e:
        return getattr(importlib.import_module("src." + module, package=None), cls)


def get_inpainting_condition(model, image, mask):
    latent_size = [x // 8 for x in image.size]
    condition_x0 = (
        model.vae.encode(image.torch().cuda() * ~mask.torch(0).bool().cuda()).mean
        * model.config.scale_factor
    )
    condition_mask = mask.resize(latent_size[::-1]).cuda().torch(0).bool().float()
    return torch.cat([condition_mask, condition_x0], 1)


def get_t2i_model(config_folder, model_folder):
    model_t2i = SimpleNamespace()

    model_t2i.config = OmegaConf.load(
        f"{config_folder}/ddpm/v1.yaml"
    )  # smplfusion.options.ddpm.v1_yaml
    model_t2i.schedule = scheduler.linear(
        model_t2i.config.timesteps,
        model_t2i.config.linear_start,
        model_t2i.config.linear_end,
    )

    model_t2i.unet = load_obj(
        OmegaConf.load(f"{config_folder}/unet/v1.yaml")
    ).cuda()  # smplfusion.options.unet.v1_yaml.cuda()
    model_t2i.vae = load_obj(
        OmegaConf.load(f"{config_folder}/vae.yaml")
    ).cuda()  # smplfusion.options.vae_yaml.cuda()
    model_t2i.encoder = load_obj(
        OmegaConf.load(f"{config_folder}/encoders/clip.yaml")
    ).cuda()  # smplfusion.options.encoders.clip_yaml.cuda()

    unet_state_dict = torch.load(
        f"{model_folder}/unet.ckpt"
    )  # assets.models.sd_1_5_inpainting.unet_ckpt
    vae_state_dict = torch.load(
        f"{model_folder}/vae.ckpt"
    )  # assets.models.sd_1_5_inpainting.vae_ckpt
    encoder_state_dict = torch.load(
        f"{model_folder}/encoder.ckpt"
    )  # assets.models.sd_1_5_inpainting.encoder_ckpt

    model_t2i.unet.load_state_dict(unet_state_dict)
    model_t2i.vae.load_state_dict(vae_state_dict)
    model_t2i.encoder.load_state_dict(encoder_state_dict, strict=False)

    model_t2i.unet = model_t2i.unet.requires_grad_(False).eval()
    model_t2i.vae = model_t2i.vae.requires_grad_(False).eval()
    model_t2i.encoder = model_t2i.encoder.requires_grad_(False).eval()
    return model_t2i, unet_state_dict


def get_inpainting_model(config_folder, model_folder):
    model_inp = SimpleNamespace()
    model_inp.config = OmegaConf.load(
        f"{config_folder}/ddpm/v1.yaml"
    )  # smplfusion.options.ddpm.v1_yaml
    model_inp.schedule = scheduler.linear(
        model_inp.config.timesteps,
        model_inp.config.linear_start,
        model_inp.config.linear_end,
    )

    model_inp.unet = load_obj(
        OmegaConf.load(f"{config_folder}/unet/inpainting/v1.yaml")
    ).cuda()  # smplfusion.options.unet.inpainting.v1_yaml.cuda()
    model_inp.vae = load_obj(
        OmegaConf.load(f"{config_folder}/vae.yaml")
    ).cuda()  # smplfusion.options.vae_yaml.cuda()
    model_inp.encoder = load_obj(
        OmegaConf.load(f"{config_folder}/encoders/clip.yaml")
    ).cuda()  # smplfusion.options.encoders.clip_yaml.cuda()

    unet_state_dict = torch.load(
        f"{model_folder}/unet.ckpt"
    )  # assets.models.sd_1_5_inpainting.unet_ckpt
    vae_state_dict = torch.load(
        f"{model_folder}/vae.ckpt"
    )  # assets.models.sd_1_5_inpainting.vae_ckpt
    encoder_state_dict = torch.load(
        f"{model_folder}/encoder.ckpt"
    )  # assets.models.sd_1_5_inpainting.encoder_ckpt

    model_inp.unet.load_state_dict(unet_state_dict)
    model_inp.vae.load_state_dict(vae_state_dict)
    model_inp.encoder.load_state_dict(encoder_state_dict, strict=False)

    model_inp.unet = model_inp.unet.requires_grad_(False).eval()
    model_inp.vae = model_inp.vae.requires_grad_(False).eval()
    model_inp.encoder = model_inp.encoder.requires_grad_(False).eval()

    return model_inp, unet_state_dict


def get_lora(lora_path):
    _lora_state_dict = safetensors.torch.load_file(lora_path)

    # Dictionary for converting
    unet_conversion_dict = {}
    unet_conversion_dict.update(convert_diffusers.unet_conversion_map_layer)
    unet_conversion_dict.update(convert_diffusers.unet_conversion_map_resnet)
    unet_conversion_dict.update(convert_diffusers.unet_conversion_map)
    unet_conversion_dict = {
        y.replace(".", "_"): x for x, y in unet_conversion_dict.items()
    }
    unet_conversion_dict["lora_unet_"] = ""

    lora_state_dict = {}
    for key in _lora_state_dict:
        key_converted = key
        for x, y in unet_conversion_dict.items():
            key_converted = key_converted.replace(x, y)
        lora_state_dict[key_converted] = _lora_state_dict[key]

    return lora_state_dict
