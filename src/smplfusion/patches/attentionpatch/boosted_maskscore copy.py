from ... import share
import torch

# import xformers
# import xformers.ops

import torchvision.transforms.functional as TF

from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat

qkv_reduce_dims = [-1]
increase_indices = [1,2]

w8 = 0.
w16 = 0.
w32 = 0.
w64 = 0.

def forward_and_save(self, x, context=None, mask=None):
    att_type = "self" if context is None else "cross"

    h = self.heads
    q = self.to_q(x)
    is_cross = context is not None
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

    scale = self.dim_head**-0.5
    sim = einsum("b i d, b j d -> b i j", q, k) * scale
    
    if is_cross:
        N = sim.shape[0] // 2
        
        delta_uncond = torch.zeros_like(sim[:N])

        if q.shape[1] == share.input_shape.res8:
            W = w8 + torch.zeros_like(sim[:N])
        if q.shape[1] == share.input_shape.res16:
            W = w16 + torch.zeros_like(sim[:N])
        if q.shape[1] == share.input_shape.res32:
            W = w32 + torch.zeros_like(sim[:N])
        if q.shape[1] == share.input_shape.res64:
            W = w64 + torch.zeros_like(sim[:N])

        tokens = torch.eye(77)[increase_indices].sum(0).cuda()

        sim[N:].argmax()

        max_sim = sim[N:].detach().amax(dim=qkv_reduce_dims, keepdim = True) # (16x1x1)
        sigma = share.schedule.sqrt_noise_signal_ratio[share.timestep]
        mask = share.input_mask.get_res(q, 'cuda').reshape(1,-1,1)

        delta_cond = W * max_sim * mask * tokens
        sim += torch.cat([delta_uncond, delta_cond])

    if hasattr(share, '_crossattn_similarity') and x.shape[1] == share.input_shape.res16 and att_type == 'cross':
        share._crossattn_similarity.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res8') and x.shape[1] == share.input_shape.res8 and att_type == 'cross':
        share._crossattn_similarity_res8.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res16') and x.shape[1] == share.input_shape.res16 and att_type == 'cross':
        share._crossattn_similarity_res16.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res32') and x.shape[1] == share.input_shape.res32 and att_type == 'cross':
        share._crossattn_similarity_res32.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res64') and x.shape[1] == share.input_shape.res64 and att_type == 'cross':
        share._crossattn_similarity_res64.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts

    sim = sim.softmax(dim=-1)
    out = einsum("b i j, b j d -> b i d", sim, v)
    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    return self.to_out(out)