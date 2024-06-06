from . import share, scheduler
from .ddim import DDIM
from .patches import router, attentionpatch, transformerpatch
from . import options
from pytorch_lightning import seed_everything
import open_clip

def count_tokens(prompt):
    tokens = open_clip.tokenize(prompt)[0]
    return (tokens > 0).sum()
    
def tokenize(prompt):
    tokens = open_clip.tokenize(prompt)[0]
    return [open_clip.tokenizer._tokenizer.decoder[x.item()] for x in tokens]

def get_token_idx(prompt, prefix, positive_prompt):
    prompt = prefix.format(prompt)
    return list(range(1 + prefix.split(' ').index('{}'), tokenize(prompt).index('<end_of_text>'))) + [tokenize(prompt + positive_prompt).index('<end_of_text>')]

def load_model_v2_inpainting(folder):
    model_config = options.ddpm.v1_yaml

    unet = options.unet.inpainting.v2_yaml.eval().cuda()
    unet.load_state_dict(folder.unet_ckpt)
    unet = unet.requires_grad_(False)

    vae = options.vae_yaml.eval().cuda()
    vae.load_state_dict(folder.vae_ckpt)
    vae = vae.requires_grad_(False)

    encoder = options.encoders.openclip_yaml.eval().cuda()
    encoder.load_state_dict(folder.encoder_ckpt)
    encoder = encoder.requires_grad_(False)
    
    return model_config, unet, vae, encoder

def load_model_v15_inpainting(folder):
    model_config = options.ddpm.v1_yaml
    
    unet = options.unet.inpainting.v1_yaml.eval().cuda()
    unet.load_state_dict(folder.unet_ckpt)
    unet = unet.requires_grad_(False)

    vae = options.vae_yaml.eval().cuda()
    vae.load_state_dict(folder.vae_ckpt)
    vae = vae.requires_grad_(False)

    encoder = options.encoders.clip_yaml.eval().cuda()
    encoder.load_state_dict(folder.encoder_ckpt)
    encoder = encoder.requires_grad_(False)

    return model_config, unet, vae, encoder

def load_model_v2(folder):
    model_config = options.ddpm.v1_yaml
    unet = options.unet.v2_yaml.eval().cuda()
    unet.load_state_dict(folder.unet_ckpt)
    unet = unet.requires_grad_(False)
    vae = options.vae_yaml.eval().cuda()
    vae.load_state_dict(folder.vae_ckpt)
    vae = vae.requires_grad_(False)
    encoder = options.encoders.openclip_yaml.eval().cuda()
    encoder.load_state_dict(folder.encoder_ckpt)
    encoder = encoder.requires_grad_(False)
    return model_config, unet, vae, encoder

def load_model_v1(folder):
    model_config = options.ddpm.v1_yaml
    unet = options.unet.v1_yaml.eval().cuda()
    unet.load_state_dict(folder.unet_ckpt)
    unet = unet.requires_grad_(False)
    vae = options.vae_yaml.eval().cuda()
    vae.load_state_dict(folder.vae_ckpt)
    vae = vae.requires_grad_(False)
    encoder = options.encoders.clip_yaml.eval().cuda()
    encoder.load_state_dict(folder.encoder_ckpt)
    encoder = encoder.requires_grad_(False)
    return model_config, unet, vae, encoder

def load_unet_v2(folder):
    unet = options.unet.v2_yaml.eval().cuda()
    unet.load_state_dict(folder.unet_ckpt)
    unet = unet.requires_grad_(False)
    return unet