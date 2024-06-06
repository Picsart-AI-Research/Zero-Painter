from ... import share
import torch

# import xformers
# import xformers.ops

import torchvision.transforms.functional as TF

from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat
import numpy as np

qkv_reduce_dims = [-1]
increase_indices = [1,2]
topk_heads = 0.5

w8 = 0.
w16 = 0.
w32 = 0.
w64 = 0.

def forward(self, x, context=None, mask=None):
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
        if share.timestep < 0:
            # Choose the weight
            if q.shape[1] == share.input_shape.res8: w = w8
            if q.shape[1] == share.input_shape.res16: w = w16
            if q.shape[1] == share.input_shape.res32: w = w32
            if q.shape[1] == share.input_shape.res64: w = w64
            mask = share.input_mask.get_res(q, 'cuda').reshape(-1)

            sim = rearrange(sim, "(b h) n d -> b h n d", h=h) # (2x10x4096x77)
            sim[1,...,0] -= 0.2 * sim[1, ..., 0] * mask

            # # Get coefficients
            # sigma = (1 - share.schedule.alphas[share.timestep])
            # sim_max = sim.detach().amax(dim=qkv_reduce_dims, keepdim = True) # (2x10x1x1)

            # # Use to modify only the conditional part
            # batch_one_hot = torch.tensor([0,1.])[:, None, None, None].cuda() # (2x1x1x1)
                
            # for token_idx in increase_indices:
            #     # Get the index of the head to be modified
            #     n_topk = int(sim.shape[1] * topk_heads)
            #     head_indices = (sim[1,...,1]).amax(dim = 1).topk(n_topk).indices.cpu()
            #     head_one_hot = torch.eye(h)[head_indices]
            #     # head_one_hot[n_topk // 2:] *= 0.5
            #     head_one_hot = head_one_hot.sum(0).cuda()[None,:,None,None] # (1x10x1x1)

            #     # head_one_hot = (1 / (1 + torch.arange(sim.shape[1]))) ** 0.5
            #     # head_one_hot = head_one_hot.cuda()[None,:,None,None]
    
            #     # Get the one hot token index
            #     token_one_hot = torch.eye(77)[token_idx].cuda()[None,None,None,:] # (1x1x1x77)

            #     sim += w * sim_max * mask * token_one_hot * head_one_hot * batch_one_hot

            sim = rearrange(sim, "b h n d -> (b h) n d", h=h) # (2x10x4096x77)
    if not is_cross and q.shape[1] == share.input_shape.res16:
        # shape of sim: 20 x 4096 x 4096
        if share.timestep < 0:
            sim_max = sim.detach().amax(dim=-1, keepdim = True) # (20x1x1)

            mask = share.input_mask.get_res(q, 'cuda').reshape(-1)
            delta = (mask[:,None] @ mask[None,:])[None]
            gamma = ((1 - mask[:,None]) @ (mask[None,:]))[None]
            
            sim = sim * (1 + 1.0 * share.schedule.sqrt_one_minus_alphas[share.timestep] * delta)
            sim = sim * (1 - 1.0 * share.schedule.sqrt_one_minus_alphas[share.timestep] * gamma)
            # sim = sim + 1.0 * share.schedule.sqrt_noise_signal_ratio[share.timestep] * sim_max * delta

    # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity') and x.shape[1] == share.input_shape.res16 and att_type == 'cross':
        share._crossattn_similarity.append(torch.stack(share.reshape(sim).chunk(2)))
    if hasattr(share, '_crossattn_similarity_res8') and x.shape[1] == share.input_shape.res8 and att_type == 'cross':
        share._crossattn_similarity_res8.append(torch.stack(share.reshape(sim).chunk(2)))
    if hasattr(share, '_crossattn_similarity_res16') and x.shape[1] == share.input_shape.res16 and att_type == 'cross':
        share._crossattn_similarity_res16.append(torch.stack(share.reshape(sim).chunk(2)))
    if hasattr(share, '_crossattn_similarity_res32') and x.shape[1] == share.input_shape.res32 and att_type == 'cross':
        share._crossattn_similarity_res32.append(torch.stack(share.reshape(sim).chunk(2)))
    if hasattr(share, '_crossattn_similarity_res64') and x.shape[1] == share.input_shape.res64 and att_type == 'cross':
        share._crossattn_similarity_res64.append(torch.stack(share.reshape(sim).chunk(2)))

    sim = sim.softmax(dim=-1)
    out = einsum("b i j, b j d -> b i d", sim, v)
    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    return self.to_out(out)

forward_and_save = forward