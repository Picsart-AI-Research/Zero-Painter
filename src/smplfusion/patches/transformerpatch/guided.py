import torch 
from ... import share

def set_wsa(value):
    global wsa
    wsa = torch.tensor([1., value])[:,None,None].cuda()
def set_wca(value):
    global wca
    wca = torch.tensor([1., value])[:,None,None].cuda()
def set_wff(value):
    global wff
    wff = torch.tensor([1., value])[:,None,None].cuda()

set_wca(1.)
set_wsa(1.)
set_wff(1.)

guidance_scale = 7.5

def forward(self, x, context=None):
    # print (x.shape)
    mask = share.input_mask.get_res(x).reshape(-1,1).cuda()
    
    _out_sa =        wsa * self.attn1(self.norm1(x), None) # Self Attn.
    
    # _out_ca_uncond = wca * self.attn2(self.norm2(x), context=context) # Cross Attn. "Unconditional"
    _out_ca        = wca * self.attn2(self.norm2(x + _out_sa), context=context) # Cross Attn.

    if share.timestep < 0:
        # x = x + (1 - mask) * _out_sa + mask * (guidance_scale * _out_ca_uncond + (1 - guidance_scale) * (_out_sa + _out_ca))
        x = x + (1 - mask) * (_out_sa + _out_ca) + mask * (guidance_scale * _out_ca_uncond + (1 - guidance_scale) * (_out_sa + _out_ca))
    else:
        x = x + _out_sa + _out_ca

    _out_ff = wff * self.ff(self.norm3(x))
    x = x + _out_ff

    if False:
        save([_out_ca, _out_sa, _out_ff])

    return x

forward_and_save = forward

def save(val):# Save outputs
    if hasattr(share, '_basic_transformer_selfattn'):
        share._basic_transformer_selfattn.append(share.reshape(val[0]))
    if hasattr(share, '_basic_transformer_crossattn'):
        share._basic_transformer_crossattn.append(share.reshape(val[1]))
    if hasattr(share, '_basic_transformer_ff'):
        share._basic_transformer_ff.append(share.reshape(val[1]))