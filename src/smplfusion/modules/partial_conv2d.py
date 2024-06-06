###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn, cuda

from .. import share

partial_res = [8, 16, 32, 64]

# class PartialConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):

#         # whether the mask is multi-channel or not
#         if 'multi_channel' in kwargs:
#             self.multi_channel = kwargs['multi_channel']
#             kwargs.pop('multi_channel')
#         else:
#             self.multi_channel = False  

#         if 'return_mask' in kwargs:
#             self.return_mask = kwargs['return_mask']
#             kwargs.pop('return_mask')
#         else:
#             self.return_mask = False

#         super(PartialConv2d, self).__init__(*args, **kwargs)

#         if self.multi_channel:
#             self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
#         else:
#             self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
#         self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

#         self.last_size = (None, None, None, None)
#         self.update_mask = None
#         self.mask_ratio = None

#     def forward(self, input, mask_in=None):
#         assert len(input.shape) == 4
#         if mask_in is not None or self.last_size != tuple(input.shape):
#             self.last_size = tuple(input.shape)

#             with torch.no_grad():
#                 if self.weight_maskUpdater.type() != input.type():
#                     self.weight_maskUpdater = self.weight_maskUpdater.to(input)

#                 if mask_in is None:
#                     # if mask is not provided, create a mask
#                     if self.multi_channel:
#                         mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
#                     else:
#                         mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
#                 else:
#                     mask = mask_in
                        
#                 self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

#                 # for mixed precision training, change 1e-8 to 1e-6
#                 self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
#                 # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
#                 self.update_mask = torch.clamp(self.update_mask, 0, 1)
#                 self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


#         raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

#         if self.bias is not None:
#             bias_view = self.bias.view(1, self.out_channels, 1, 1)
#             output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
#             output = torch.mul(output, self.update_mask)
#         else:
#             output = torch.mul(raw_out, self.mask_ratio)


#         if self.return_mask:
#             return output, self.update_mask
#         else:
#             return output
        
        
class PartialConv2d(nn.Conv2d):
    """
    NOTE: You need to use share.set_mask(original_mask) before inferencing with PartialConv2d.
    """
    
    def __init__(self, *args, **kwargs):
        super(PartialConv2d, self).__init__(*args, **kwargs)
        
        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

    def forward(self, input):
        if share.input_mask is None:
            raise Exception('Please set share.set_mask(original_mask) before inferencing with PartialConv2d.')
        
        bias_view = torch.zeros((1, self.out_channels, 1, 1), dtype=input.dtype, device=input.device)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
        
        # Get the resized mask for the current resolution
        res = max(input.shape[2:])
        
        if res == max(share.input_mask.shape64):
            mask = share.input_mask.val64
            mask_down = share.input_mask.val32
        elif res == max(share.input_mask.shape32):
            mask = share.input_mask.val32
            mask_down = share.input_mask.val16
        elif res == max(share.input_mask.shape16):
            mask = share.input_mask.val16
            mask_down = share.input_mask.val8
        elif res == max(share.input_mask.shape8):
            mask = share.input_mask.val8
        
        mask = mask.to(input.device)
        
        # Separately perform the convolution operation on masked and known regions
        masked_input = torch.mul(input, mask)
        known_input = torch.mul(input, 1-mask)
        
        input_out =super(PartialConv2d, self).forward(input)
        masked_out = super(PartialConv2d, self).forward(masked_input)
        known_out = super(PartialConv2d, self).forward(known_input)
        
        # Calculate the rescaling weights for known and unknown regions
        
        # ############ Weighting strategy No 1 #################
        
        # pixel_counts = F.conv2d(
        #     F.pad(mask, [*self.padding, *self.padding], mode='reflect'),
        #     self.weight_maskUpdater.to(mask.device),
        #     bias=None,
        #     stride=self.stride,
        #     padding=(0, 0),
        #     dilation=self.dilation,
        #     groups=1
        # )
        
        # mask_ratio_unknown = self.slide_winsize/ (pixel_counts)
        # mask_ratio_known = self.slide_winsize / (self.slide_winsize - pixel_counts)
        
        # ################## End of No 1 #########################
        
        # ################ Weighting strategy No 2 ###############
        
        # ones_input = torch.ones_like(input)
        # masks_input = mask.repeat(1, ones_input.shape[1], 1, 1)
        
        # sum_overall = F.conv2d(
        #     F.pad(ones_input, [*self.padding, *self.padding], mode='reflect'),
        #     torch.abs(self.weight),
        #     bias=None,
        #     stride=self.stride,
        #     padding=(0, 0),
        #     dilation=self.dilation,
        #     groups=1
        # )
        
        # sum_masked = F.conv2d(
        #     F.pad(masks_input, [*self.padding, *self.padding], mode='reflect'),
        #     torch.abs(self.weight),
        #     bias=None,
        #     stride=self.stride,
        #     padding=(0, 0),
        #     dilation=self.dilation,
        #     groups=1
        # )

        # mask_ratio_unknown = sum_overall / (sum_masked)
        # mask_ratio_known = sum_overall / (sum_overall - sum_masked)
        
        # ################## End of No 2 #########################
        
        ################ Weighting strategy No 3 ###############
        if res not in partial_res:
            return input_out
        
        input_norm = torch.norm(input_out - bias_view, dim=1, keepdim=True)
        known_norm = torch.norm(known_out - bias_view, dim=1, keepdim=True)
        masked_norm = torch.norm(masked_out - bias_view, dim=1, keepdim=True)

        mask_ratio_unknown = input_norm / masked_norm
        mask_ratio_known = input_norm / known_norm
        
        ################## End of No 3 #########################
        
        # Replace nan and inf with 0.0
        mask_ratio_unknown = torch.nan_to_num(mask_ratio_unknown, nan=0.0, posinf=0.0, neginf=0.0)
        mask_ratio_known = torch.nan_to_num(mask_ratio_known, nan=0.0, posinf=0.0, neginf=0.0)
        
        ###################### DEBUG ############################
        # if res == 8:
        #     print(mask_ratio_unknown[0][0], mask_ratio_unknown.shape)
        #     print(mask_ratio_known[0][0], mask_ratio_known.shape)
        
        ################### END OF DEBUG ########################
        
        # If set to true, doesn't rescale the convolution outputs
        ignore_mask_ratio = False
        if ignore_mask_ratio:
            mask_ratio_known = 1.0
            mask_ratio_unknown = 1.0
        
        known_out = known_out - bias_view
        masked_out = masked_out - bias_view
        
        if max(self.stride) > 1:
            mask_down = mask_down.to(input.device)
            out = masked_out * mask_down * mask_ratio_unknown + known_out * (1-mask_down) * mask_ratio_known
        else: 
            out = masked_out * mask * mask_ratio_unknown + known_out * (1-mask) * mask_ratio_known        
        
        out = out + bias_view
        
        return out