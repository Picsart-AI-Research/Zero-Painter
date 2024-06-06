from ... import share
import torch

# import xformers
# import xformers.ops

import torchvision.transforms.functional as TF

from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat

qkv_reduce_dims = [-1, -2]

def forward(self, x, context=None, mask=None):
    h = self.heads
    q = self.to_q(x)
    is_cross = context is not None
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
    scale = self.dim_head**-0.5
    sim = einsum("b i d, b j d -> b i j", q, k) 
    steps_40 = [999, 974, 949, 924, 899, 874, 849, 824, 799, 774, 749, 724, 699, 674, 649, 624, 599, 574, 549, 524, 499, 474, 449, 424, 399, 374, 349, 324, 299, 274, 249, 224, 199, 174, 149, 124, 99, 74, 49, 24]
    test_ind = steps_40.index(share.timestep)
    if hasattr(share, 'list_of_masks') and is_cross:
        if q.shape[1] in [share.input_shape.res16, share.input_shape.res32] or True:
            for masked_object in share.list_of_masks:
                # sim: (16x4096x77); mask: (64x64) -> mask.reshape(-1): (4096)
                zp_condition = (
                    masked_object['w']
                    # * share.zp_sigmas[share.timestep]
                    * share.zp_sigmas[test_ind]
                    * masked_object['mask'].get_res(q, 'cuda').reshape(1,-1,1)# (1,4096,1)
                    * sim.amax(dim=qkv_reduce_dims, keepdim = True) # (16x1x1)
                ) * (torch.eye(77)[masked_object['token_idx']].sum(0).cuda()) # (1,1,77)

                zp_zeros = torch.zeros_like(zp_condition).cuda()
                final_condition = torch.concat([zp_zeros[:8,:,:],zp_condition[8:,:,:]])
                
                sim = sim+final_condition
                # print(sim.shape,"wwwww")
    del q, k

    sim = sim*scale
    sim = sim.softmax(dim=-1)

    out = einsum("b i j, b j d -> b i d", sim, v)
    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    return self.to_out(out)