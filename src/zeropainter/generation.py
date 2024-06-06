from .models import get_inpainting_condition

import torch
from src.smplfusion import share
from src.smplfusion.patches import router
from src.smplfusion.common import get_token_idx
from pytorch_lightning import seed_everything # TODO: remove
from src.smplfusion import IImage # TODO: replace with PIL.Image ?maybe
import numpy as np
from tqdm import tqdm

from torch import nn, einsum
from einops import rearrange, repeat

qkv_reduce_dims = [-1, -2]
attn_res_mask = [0, 1, 2, 3]
latent_color = "black"

negative_prompt = ""
positive_prompt = ""
negative_prompt = "worst quality, ugly, gross, disfigured, deformed, dehydrated, extra limbs, fused body parts, mutilated, malformed, mutated, bad anatomy, bad proportions, low quality, cropped, low resolution, out of frame, poorly drawn, text, watermark, letters, jpeg artifacts"
positive_prompt = ", realistic, HD, Full HD, 4K, high quality, high resolution, masterpiece, trending on artstation, realistic lighting"


class ZeroPainterForward:
    def __init__(self, masks, sigmas):
        self.sigmas = sigmas
        self.masks = masks

    def __call__(self, module, x, context=None, mask=None):
        att_type = "self" if context is None else "cross"
        context = x if context is None else context
        batch_size = x.shape[0]

        q = module.to_q(x)
        k = module.to_k(context)
        v = module.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=module.heads), (q, k, v)
        )
        scale = module.dim_head**-0.5
        sim = einsum("b i d, b j d -> b i j", q, k)

        if att_type == "cross":
            att_res_list = [
                share.input_shape.res8,
                share.input_shape.res16,
                share.input_shape.res32,
                share.input_shape.res64,
            ]
            att_res_list = [att_res_list[i] for i in attn_res_mask]

            if q.shape[1] in att_res_list:
                # for masked_object in self.masks:
                # sim: (16x4096x77); mask: (64x64) -> mask.reshape(-1): (4096)
                zp_condition = (
                    (
                        self.masks.w_positive
                        * self.masks.modified_mask.get_res(q, "cuda").reshape(
                            1, -1, 1
                        )  # (1,4096,1)
                        * torch.eye(77)[self.masks.modified_indexses_of_prompt]
                        .sum(0)
                        .cuda()  # (1,1,77)
                        + self.masks.w_negative
                        * self.masks.inverse_mask.get_res(q, "cuda").reshape(
                            1, -1, 1
                        )  # (1,4096,1)
                        * torch.eye(77)[self.masks.sot_index].sum(0).cuda()  # (1,1,77)
                    )
                    * self.sigmas[share.timestep]
                    * sim.amax(dim=qkv_reduce_dims, keepdim=True)
                )  # (16x1x1)

                zp_zeros = torch.zeros_like(zp_condition).cuda()

                if batch_size == 1:
                    final_condition = zp_condition
                elif batch_size == 2:
                    final_condition = torch.concat(
                        [
                            zp_zeros[: module.heads, :, :],
                            zp_condition[module.heads :, :, :],
                        ]
                    )
                else:
                    raise NotImplementedError(
                        "Batch sizes of larger than 1 not yet supported!"
                    )
                sim = sim + final_condition
        elif att_type == "self":
            pass
        del q, k

        sim = sim * scale
        sim = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=module.heads)
        return module.to_out(out)


def get_sigmas():
    np.random.seed(0)
    lognorm = np.concatenate(
        (np.random.lognormal(-1.2, 2, 200), np.random.lognormal(-1.2, 1.2, 800))
    )

    lognorm = lognorm.astype("float32")
    test_array = np.sort(lognorm)
    test_value = test_array
    test_value = torch.tensor([test_value]).cuda()

    res = torch.log(test_value[0] + 1)
    return res


def gen_single_object(
    model_t2i,
    zp_masks,
    my_seed_gen,
    prefix="realistic photo of a {}",
    t_max=799,
    dt=20,
    guidance_scale=7.5,
    use_lcm_multistep=False,
):
    seed_everything(my_seed_gen)

    # Setup the code to use zp
    image_w = zp_masks.image_w
    image_h = zp_masks.image_h
    prompt = zp_masks.local_prompt

    zp_masks.modified_indexses_of_prompt = get_token_idx(
        prompt, prefix, positive_prompt
    )
    # print (tokenize(prefix.format(prompt) + positive_prompt), zp_masks[0]['token_idx'])

    router.attention_forward = ZeroPainterForward(zp_masks, get_sigmas())

    share.input_shape = share.InputShape([image_w, image_h])
    latent_w, latent_h = image_w // 8, image_h // 8

    mask = zp_masks.modified_mask
    mask_test = mask.img64.torch().cuda()

    # In case we use inpainting model
    image_bg = IImage(torch.zeros(1, 3, image_h, image_w)).cuda()
    condition = get_inpainting_condition(
        model_t2i, image_bg, IImage(torch.ones_like(mask.img.torch()))
    )

    context = model_t2i.encoder.encode(
        [negative_prompt, prefix.format(prompt) + positive_prompt]
    )  # Unconditional first, then conditional
    eps = torch.randn((1, 4, latent_h, latent_w)).cuda()

    if latent_color == "black":
        latent_bg = (
            model_t2i.vae.encode(-torch.ones(1, 3, 512, 512).cuda()).mean
            * model_t2i.config.scale_factor
        )
    elif latent_color == "pink":
        latent_bg = -1
    elif latent_color == "brown":
        latent_bg = 0
    elif latent_color == "grey":
        latent_bg = (
            model_t2i.vae.encode(torch.zeros(1, 3, 512, 512).cuda()).mean
            * model_t2i.config.scale_factor
        )
    elif latent_color == "green":
        latent_bg = (
            model_t2i.vae.encode(
                (
                    torch.eye(3)[1][None, :, None, None] + torch.zeros(1, 3, 512, 512)
                ).cuda()
            ).mean
            * model_t2i.config.scale_factor
        )

    zT = (
        model_t2i.schedule.sqrt_alphas[t_max] * latent_bg
        + model_t2i.schedule.sqrt_one_minus_alphas[t_max] * eps
    )
    zT = (mask_test < 0) * zT + (mask_test > 0) * eps

    timesteps = list(range(t_max, 0, -dt))
    n_empty = len(list(range(999, t_max, -dt)))
    z0_list, zt_list, eps_list = [None] * n_empty, [None] * n_empty, [None] * n_empty
    with torch.autocast("cuda"), torch.no_grad():
        zt = zT

        for index, t in enumerate(tqdm(timesteps)):
            # _zt = torch.cat([zt, condition], 1) # In case we use inpainting model
            _zt = zt
            share.timestep = t  # Set the current timestep

            if use_lcm_multistep:
                eps = model_t2i.unet(
                    _zt, timesteps=torch.tensor([t]).cuda(), context=context[1][None]
                )
            else:
                eps_uncond, eps = model_t2i.unet(
                    torch.cat([_zt, _zt]),
                    timesteps=torch.tensor([t, t]).cuda(),
                    context=context,
                ).chunk(2)
                eps = eps_uncond + guidance_scale * (eps - eps_uncond)

            z0 = (
                zt - model_t2i.schedule.sqrt_one_minus_alphas[t] * eps
            ) / model_t2i.schedule.sqrt_alphas[
                t
            ]  # This is the predicted output image at step t [pred_x0]
            zt = (
                model_t2i.schedule.sqrt_alphas[t - dt] * z0
                + model_t2i.schedule.sqrt_one_minus_alphas[t - dt] * eps
            )  # This is the z latent for the next step [x_prev]

            if t > 250:
                # latent_bg = schedule.sqrt_alphas[t] * mask_test + schedule.sqrt_one_minus_alphas[t] * torch.randn((1,4,latent_h,latent_w)).cuda()
                # latent_bg = schedule.sqrt_alphas[t] * black_latent + schedule.sqrt_one_minus_alphas[t] * torch.randn((1,4,latent_h,latent_w)).cuda()
                # latent_bg = schedule.sqrt_one_minus_alphas[t] * torch.randn((1,4,latent_h,latent_w)).cuda()
                latent_bg_t = (
                    model_t2i.schedule.sqrt_alphas[t] * latent_bg
                    + model_t2i.schedule.sqrt_one_minus_alphas[t]
                    * torch.randn((1, 4, latent_h, latent_w)).cuda()
                )
                zt = (mask_test < 0) * latent_bg_t + (mask_test > 0) * zt

            eps_list.append(eps)
            z0_list.append(z0)
            zt_list.append(zt)

    return eps_list, z0_list, zt_list


def gen_batch_object(
    model_t2i, zp_masks, my_seed_gen, prefix="realistic photo of a {}"
):
    seed_everything(my_seed_gen)

    # Setup the code to use zp

    image_w = zp_masks[0]["image_w"]
    image_h = zp_masks[0]["image_h"]
    prompt = zp_masks[0]["prompt"]

    zp_masks[0]["token_idx"] = get_token_idx(prompt, prefix, positive_prompt)
    # print (tokenize(prefix.format(prompt) + positive_prompt), zp_masks[0]['token_idx'])

    router.attention_forward = ZeroPainterForward(zp_masks, get_sigmas())

    share.input_shape = share.InputShape([image_w, image_h])
    latent_w, latent_h = image_w // 8, image_h // 8

    mask = zp_masks[0]["mask"]
    mask_test = mask.img64.torch().cuda()

    dt = 20  # NUmber of steps = 1000 / dt

    # In case we use inpainting model
    image_bg = IImage(torch.zeros(1, 3, image_h, image_w)).cuda()
    condition = get_inpainting_condition(
        model_t2i, image_bg, IImage(torch.ones_like(mask.img.torch()))
    )

    context = model_t2i.encoder.encode(
        [negative_prompt, prefix.format(prompt) + positive_prompt]
    )  # Unconditional first, then conditional
    eps = torch.randn((1, 4, latent_h, latent_w)).cuda()

    if latent_color == "black":
        latent_bg = (
            model_t2i.vae.encode(-torch.ones(1, 3, 512, 512).cuda()).mean
            * model_t2i.config.scale_factor
        )
    elif latent_color == "pink":
        latent_bg = -1
    elif latent_color == "brown":
        latent_bg = 0
    elif latent_color == "grey":
        latent_bg = (
            model_t2i.vae.encode(torch.zeros(1, 3, 512, 512).cuda()).mean
            * model_t2i.config.scale_factor
        )
    elif latent_color == "green":
        latent_bg = (
            model_t2i.vae.encode(
                (
                    torch.eye(3)[1][None, :, None, None] + torch.zeros(1, 3, 512, 512)
                ).cuda()
            ).mean
            * model_t2i.config.scale_factor
        )

    zT = (
        model_t2i.schedule.sqrt_alphas[799] * latent_bg
        + model_t2i.schedule.sqrt_one_minus_alphas[799] * eps
    )
    zT = (mask_test < 0) * zT + (mask_test > 0) * eps

    timesteps = list(range(799, 0, -dt))
    n_empty = len(list(range(999, 799, -dt)))
    z0_list, zt_list, eps_list = [None] * n_empty, [None] * n_empty, [None] * n_empty
    with torch.autocast("cuda"), torch.no_grad():
        zt = zT

        for index, t in enumerate(tqdm(timesteps)):
            # _zt = torch.cat([zt, condition], 1) # In case we use inpainting model
            _zt = zt

            share.timestep = t  # Set the current timestep
            eps_uncond, eps = model_t2i.unet(
                torch.cat([_zt, _zt]),
                timesteps=torch.tensor([t, t]).cuda(),
                context=context,
            ).chunk(2)

            eps = eps_uncond + 7.5 * (eps - eps_uncond)
            z0 = (
                zt - model_t2i.schedule.sqrt_one_minus_alphas[t] * eps
            ) / model_t2i.schedule.sqrt_alphas[
                t
            ]  # This is the predicted output image at step t [pred_x0]
            zt = (
                model_t2i.schedule.sqrt_alphas[t - dt] * z0
                + model_t2i.schedule.sqrt_one_minus_alphas[t - dt] * eps
            )  # This is the z latent for the next step [x_prev]

            if t > 250:
                # latent_bg = schedule.sqrt_alphas[t] * mask_test + schedule.sqrt_one_minus_alphas[t] * torch.randn((1,4,latent_h,latent_w)).cuda()
                # latent_bg = schedule.sqrt_alphas[t] * black_latent + schedule.sqrt_one_minus_alphas[t] * torch.randn((1,4,latent_h,latent_w)).cuda()
                # latent_bg = schedule.sqrt_one_minus_alphas[t] * torch.randn((1,4,latent_h,latent_w)).cuda()
                latent_bg_t = (
                    model_t2i.schedule.sqrt_alphas[t] * latent_bg
                    + model_t2i.schedule.sqrt_one_minus_alphas[t]
                    * torch.randn((1, 4, latent_h, latent_w)).cuda()
                )
                zt = (mask_test < 0) * latent_bg_t + (mask_test > 0) * zt

            eps_list.append(eps)
            z0_list.append(z0)
            zt_list.append(zt)

    return eps_list, z0_list, zt_list
