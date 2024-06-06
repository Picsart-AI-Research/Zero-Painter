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

# set_wca(1.)
# set_wsa(1.)
# set_wff(1.)

wca = 1.0
wsa = 1.0
wff = 1.0

def forward(self, x, context=None):
    x = x + wsa * self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) # Self Attn.
    x = x + wca * self.attn2(self.norm2(x), context=context) # Cross Attn.
    x = x + wff * self.ff(self.norm3(x))
    return x

def forward_and_save(self, x, context=None):
    val = [x]
    val.append(wsa * self.attn1(self.norm1(x), context=context if self.disable_self_attn else None)) # Self Attn.
    x = x + val[-1]
    val.append(wca * self.attn2(self.norm2(x), context=context)) # Cross Attn.
    x = x + val[-1]
    val.append(wff * self.ff(self.norm3(x)))
    x = x + val[-1]
    
    # Save outputs
    if hasattr(share, '_basic_transformer_input') and x.shape[1] == share.input_shape.res16:
        share._basic_transformer_input.append(share.reshape(val[0]))
    if hasattr(share, '_basic_transformer_selfattn') and x.shape[1] == share.input_shape.res16:
        share._basic_transformer_selfattn.append(share.reshape(val[1]))
    if hasattr(share, '_basic_transformer_crossattn') and x.shape[1] == share.input_shape.res16:
        share._basic_transformer_crossattn.append(share.reshape(val[2]))
    if hasattr(share, '_basic_transformer_ff') and x.shape[1] == share.input_shape.res16:
        share._basic_transformer_ff.append(share.reshape(val[3]))
    return x
