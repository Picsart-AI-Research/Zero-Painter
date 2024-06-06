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

def forward(self, x, context=None):
    x = x + wca * self.attn2(self.norm2(x), context=context) # Cross Attn.
    x = x + wsa * self.attn1(self.norm1(x), None) # Self Attn.
    x = x + wff * self.ff(self.norm3(x))
    return x

def forward_and_save(self, x, context=None):
    val = [x]
    val.append(wsa * self.attn1(self.norm1(x), None)) # Self Attn.
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


# ========= MODIFICATIONS ============= #


def forward_and_save3(self, x, context=None):
    val = []
    val.append(self.attn1(self.norm1(x), None)) # Self Attn.
    modify_res = [share.input_mask.res16, share.input_mask.res32, share.input_mask.res64]

    if x.shape[1] in modify_res:
        val[-1] = val[-1] + (wsa - 1) * share.input_mask.get_res(x).reshape(-1)[:,None].cuda() * val[-1]
    x = x +  val[-1]
    
    val.append(self.attn2(self.norm2(x), context=context)) # Cross Attn.

    if x.shape[1] in modify_res:
        val[-1] = val[-1] + (wca - 1) * share.input_mask.get_res(x).reshape(-1)[:,None].cuda() * val[-1]
    x = x + val[-1]
    
    val.append(self.ff(self.norm3(x)))
    
    if x.shape[1] in modify_res:
        val[-1] = val[-1] + (wff - 1) * share.input_mask.get_res(x).reshape(-1)[:,None].cuda() * val[-1]
    x = x + val[-1]
    if hasattr(share, 'out_basic_transformer_block') and x.shape[1] == share.input_mask.res16:
        share.out_basic_transformer_block.append(val)
    return x

def forward_and_save2(self, x, context=None):
    val = []
    val.append(self.attn1(self.norm1(x), None)) # Self Attn.
    x = x + wsa * val[-1]
    val.append(self.attn2(self.norm2(x), context=context)) # Cross Attn.
    print (val[-1].shape, share.input_mask.val16.shape)
    x = x + val[-1] + wca * share.input_mask.val16 * val[-1]
    val.append(self.ff(self.norm3(x)))
    x = x + wff * val[-1]
    if hasattr(share, 'out_basic_transformer_block') and x.shape[1] == share.input_mask.res16:
        share.out_basic_transformer_block.append(val)
    return x

def forward_and_reweight(self, x, context=None):
    modify_res = [share.input_mask.res16, share.input_mask.res32, share.input_mask.res64]

    _attn1 = self.attn1(self.norm1(x), None) # Self Attn.
    _attn2 = self.attn2(self.norm2(x + _attn1), context=context) # Cross Attn.
    _ff = self.ff(self.norm3(x + _attn1 + _attn2))

    if x.shape[1] in modify_res:
        lm1,lm2,lm3 = wsa - 1, wca - 1, wff - 1
        _attn1 *= (1 + lm1 * share.input_mask.get_res(x).reshape(-1)[:,None].cuda())
        _attn2 *= (1 + lm2 * share.input_mask.get_res(x).reshape(-1)[:,None].cuda())
        _ff    *= (1 + lm3 * share.input_mask.get_res(x).reshape(-1)[:,None].cuda())

    if hasattr(share, 'out_basic_transformer_block') and x.shape[1] == share.input_mask.res16:
        share.out_basic_transformer_block.append([_attn1, _attn2, _ff])

    return x + _attn1 + _attn2 + _ff
    # return x + (w_sa * _attn1 + w_ca * _attn2 + w_ff * _ff)
    # return x + (w_sa * _attn1 + w_ca * _attn2 + w_ff * _ff) / ((w_sa + w_ca + w_ff) / 3)

def forward_and_reweight2(self, x, context=None):
    _attn1 = self.attn1(self.norm1(x), None) # Self Attn.
    _attn2 = self.attn2(self.norm2(x + _attn1), context=context) # Cross Attn.
    _ff = self.ff(self.norm3(x + _attn1 + _attn2))

    _attn1 += (wsa - 1) / ((wsa + wca + wff) / 3) * share.input_mask.get_res(x).reshape(-1)[:,None].cuda() * _attn1
    _attn2 += (wca - 1) / ((wsa + wca + wff) / 3) * share.input_mask.get_res(x).reshape(-1)[:,None].cuda() * _attn2
    _ff    += (wff - 1) / ((wsa + wca + wff) / 3) * share.input_mask.get_res(x).reshape(-1)[:,None].cuda() * _ff


    if hasattr(share, 'out_basic_transformer_block') and x.shape[1] == share.input_mask.res16:
        share.out_basic_transformer_block.append([_attn1, _attn2, _ff])
        # share.out_basic_transformer_block.append([
        #     w_sa / ((w_sa + w_ca + w_ff) / 3) * _attn1, 
        #     w_ca / ((w_sa + w_ca + w_ff) / 3) * _attn2, 
        #     w_ff / ((w_sa + w_ca + w_ff) / 3) * _ff
        # ])

    return x + _attn1 + _attn2 + _ff
    # return x + (w_sa * _attn1 + w_ca * _attn2 + w_ff * _ff)
    # return x + (w_sa * _attn1 + w_ca * _attn2 + w_ff * _ff) / ((w_sa + w_ca + w_ff) / 3)