import torch
from torch import nn, einsum
from einops import rearrange, repeat
from ... import share
from ..attentionpatch import introvert

w_ff = 1.
w_sa = 1.
w_ca = 1.

use_grad = True


def forward(self, x, context=None):
    # with torch.no_grad():
    if use_grad:
        y, self_v, self_sim = self.attn1(self.norm1(x), None) # Self Attn.
        
        x_uncond, x_cond = x.chunk(2)
        context_uncond, context_cond = context.chunk(2)
        
        y_uncond, y_cond = y.chunk(2)
        self_sim_uncond, self_sim_cond = self_sim.chunk(2)
        self_v_uncond, self_v_cond = self_v.chunk(2)

        # Calculate CA similarities with conditional context
        cross_h = self.attn2.heads
        cross_q = self.attn2.to_q(self.norm2(x_cond+y_cond))
        cross_k = self.attn2.to_k(context_cond)
        cross_v = self.attn2.to_v(context_cond)

        cross_q, cross_k, cross_v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=cross_h), (cross_q, cross_k, cross_v))

        with torch.autocast(enabled=False, device_type = 'cuda'):
            cross_q, cross_k = cross_q.float(), cross_k.float()
            cross_sim = einsum('b i d, b j d -> b i j', cross_q, cross_k) * self.attn2.scale
        
        del cross_q, cross_k
        cross_sim = cross_sim.softmax(dim=-1) # Up to this point cross_sim is regular cross_sim in CA layer

        cross_sim = cross_sim.mean(dim=0) # Calculate mean across heads
        
    # Introvert Attention rescale heppens here
    y_cond = introvert.introvert_rescale(
        y_cond, self_v_cond, self_sim_cond, cross_sim, self.attn1.heads, self.attn1.to_out) # Rescale cond
    y_uncond = introvert.introvert_rescale(
        y_uncond, self_v_uncond, self_sim_uncond, cross_sim, self.attn1.heads, self.attn1.to_out) # Rescale uncond
    
    y = torch.cat([y_uncond, y_cond], dim=0)
    
    x = x + w_sa * y
    x = x + w_ca * self.attn2(self.norm2(x), context=context) # Cross Attn.
    x = x + w_ff * self.ff(self.norm3(x))
    return x