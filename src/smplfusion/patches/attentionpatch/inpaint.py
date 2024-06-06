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

def forward(self, x, context=None, mask=None):
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
        if q.shape[1] in [share.input_shape.res16]:
            # For simplicity, assume token 1 is target, token 2 is <EOT> (1 word label)
            # sim: (16x4096x77); mask: (64x64) -> mask.reshape(-1): (4096)
            N = sim.shape[0]//2
            sim[N:] += (
                  share.w 
                * share.noise_signal_ratio[share.timestep]
                * share.input_mask.get_res(q, 'cuda').reshape(1,-1,1)# (1,4096,1)
                * sim[N:].amax(dim=qkv_reduce_dims, keepdim = True) # (16x1x1)
            ) * (torch.eye(77)[increase_indices].sum(0).cuda()) # (1,1,77)

    del q, k
    sim = sim.softmax(dim=-1)

    out = einsum("b i j, b j d -> b i d", sim, v)
    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    return self.to_out(out)