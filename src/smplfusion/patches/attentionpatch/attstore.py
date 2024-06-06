# CrossAttn precision handling
import os
import torch
from torch import nn

from torch import einsum
from einops import rearrange, repeat
from ... import share

att_res = [16 * 16]
force_idx = 1

def forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    att_type = "self" if context is None else "cross"
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

    sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
    
    if att_type == "cross" and q.shape[1] in att_res:
        share.sim.append(sim)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum("b i j, b j d -> b i d", sim, v)
    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    return self.to_out(out)

def forward_force(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    att_type = "self" if context is None else "cross"
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

    sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
    
    if att_type == "cross":
        context_dim = context.shape[1] # Number of tokens, might not be 77
        
        # if q.shape[1] == share.input_mask.res64:
        #     _sim = 100 * torch.eye(context_dim)[force_idx].half().cuda()[None].repeat(sim.shape[0], sim.shape[1], 1)
        #     _sim += 100 * torch.eye(context_dim)[0].half().cuda()[None].repeat(sim.shape[0], sim.shape[1], 1)
        #     sim[:,share.input_mask.val64.reshape(-1) > 0,:] = _sim[:,share.input_mask.val64.reshape(-1) > 0,:]
        # if q.shape[1] == share.input_mask.res32:
        #     _sim = 100 * torch.eye(context_dim)[force_idx].half().cuda()[None].repeat(sim.shape[0], sim.shape[1], 1)
        #     _sim += 100 * torch.eye(context_dim)[0].half().cuda()[None].repeat(sim.shape[0], sim.shape[1], 1)
        #     sim[:,share.input_mask.val32.reshape(-1) > 0,:] = _sim[:,share.input_mask.val32.reshape(-1) > 0,:]
        if q.shape[1] == share.input_mask.res16:
            # print (sim.shape)
            _sim = 100 * torch.eye(context_dim)[force_idx].half().cuda()[None].repeat(sim.shape[0]//2, sim.shape[1], 1)
            _sim += 100 * torch.eye(context_dim)[0].half().cuda()[None].repeat(sim.shape[0]//2, sim.shape[1], 1)
            # sim[sim.shape[0]//2:,share.input_mask.val16.reshape(-1) > 0,:] = _sim[:,share.input_mask.val16.reshape(-1) > 0,:]
            # sim[sim.shape[0]//2:, share.input_mask.val16.reshape(-1) > 0, 0] *= 0.8
            share.sim.append(sim[sim.shape[0]//2:])
    
    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum("b i j, b j d -> b i d", sim, v)
    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    if q.shape[1] == share.input_mask.res16:
        if att_type == "cross":
            share.cross_out.append(out.detach())
        if att_type == "self":
            share.self_out.append(out.detach())

    return self.to_out(out)
