from ... import share

# import xformers
# import xformers.ops

import torch
from torch import nn, einsum
import torchvision.transforms.functional as TF
from einops import rearrange, repeat

layer_mask = share.LayerMask()

def forward(self, x, context=None, mask=None):
    att_type = "self" if context is None else "cross"

    h = self.heads
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

    sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)
    out = einsum("b i j, b j d -> b i d", sim, v)

    if att_type == 'self' and q.shape[1] in [share.input_shape.res16]:
        out = out[:,torch.randperm(out.shape[1]),:]

    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    return self.to_out(out)