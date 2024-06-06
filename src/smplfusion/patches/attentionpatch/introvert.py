import cv2
import math
import numbers
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, einsum
from einops import rearrange, repeat

from ... import share
from ...libimage import IImage


# Default for version 4
introvert_res = [16, 32, 64]
introvert_on = True
token_idx = [1,2]

# Visualization purpose
viz_image = None
viz_mask = None

video_frames_selfattn = []
video_frames_crossattn = []
visualize_resolution = 16
visualize_selfattn = False
visualize_crossattn = False

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups, padding='same')


def forward(self, x, context=None, mask=None):
    is_cross = context is not None
    att_type = "self" if context is None else "cross"

    h = self.heads

    q = self.to_q(x)
    context =  x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    sim_before = sim
    del q, k

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    if hasattr(share, '_crossattn_similarity_res8') and x.shape[1] == share.input_shape.res8 and att_type == 'cross':
        share._crossattn_similarity_res8.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res16') and x.shape[1] == share.input_shape.res16 and att_type == 'cross':
        share._crossattn_similarity_res16.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res32') and x.shape[1] == share.input_shape.res32 and att_type == 'cross':
        share._crossattn_similarity_res32.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res64') and x.shape[1] == share.input_shape.res64 and att_type == 'cross':
        share._crossattn_similarity_res64.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts

    sim = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    
    if is_cross:
        return self.to_out(out)
    
    return self.to_out(out), v, sim_before


def introvert_rescale(y, self_v, self_sim, cross_sim, self_h, to_out):   
    mask = share.introvert_mask.get_res(self_v)
    shape = share.introvert_mask.get_shape(self_v)
    res = share.introvert_mask.get_res_val(self_v)
    # print (res, shape)

    ################# Introvert Attention V4 ################
    # Use this with 50% of DDIM steps
    
    # TODO: LOOK INTO THIS. WHY WITHOUT BINARY WORKS BETTER????
    # mask = (mask > 0.5).to(torch.float32)
    m = mask.to(self_v.device)
    # mask_smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).cuda()
    # m = mask_smoothing(m) # Smoothing on binary mask also works
    m = rearrange(m, 'b c h w -> b (h w) c').contiguous()
    mo = m
    m = torch.matmul(m, m.permute(0, 2, 1)) + (1-m) 
    
    cross_sim = cross_sim[:, token_idx].sum(dim=1)
    cross_sim = cross_sim.reshape(shape)
    # TODO: comment this out if it is not neccessary
    heatmap_smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).cuda()
    cross_sim = heatmap_smoothing(cross_sim.unsqueeze(0))[0]
    cross_sim = cross_sim.reshape(-1)
    cross_sim = ((cross_sim - torch.median(cross_sim.ravel())) / torch.max(cross_sim.ravel())).clip(0, 1)
  
    # If introvert attention is off, return original SA result
    if introvert_on and res in introvert_res:
        w = (1 - m) * cross_sim.reshape(1, 1, -1) + m
        # On 64 resolution make scaling with constant, as cross_sim doesn't contain semantic meaning
        if res == 64: w = m
        self_sim = self_sim * w
        self_sim_viz = self_sim # Keep for viz purposes
        self_sim = self_sim.softmax(dim=-1)        
        out = einsum('b i j, b j d -> b i d', self_sim, self_v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self_h)
        out = to_out(out)
    else: 
        self_sim_viz = self_sim # Keep for viz purposes
        out = y
    ################## END OF Introvert Attention V4 ###########################
    
    
    ################# VISUALIZE CROSS ATTENTION ###############################
    if visualize_crossattn and res == visualize_resolution:
        cross_vis = cross_sim.reshape(shape)
        up = (64 // res) * 8
        if viz_image is not None:
            heatmap = IImage(cross_vis, vmin=0).heatmap((cross_vis.shape[0]*up, cross_vis.shape[1]*up))
            video_frames_crossattn.append(viz_image + heatmap)
        else:
            heatmap = IImage(cross_vis, vmin=0).heatmap((cross_vis.shape[0]*up, cross_vis.shape[1]*up))
            video_frames_crossattn.append(heatmap)
    ###############################
    
    ################# VISUALIZE SELF ATTENTION ###############################
    if visualize_selfattn and res == visualize_resolution:
        selected = []
        up = (64 // res) * 8
        for i in range(mo.shape[1]):
            if mo[0, i, 0]:
                selected.append(self_sim_viz[:, i, :])
        selected = torch.stack(selected, dim=1)
        selected_vis = selected.mean(0).mean(0)
        
        selected_vis = selected_vis.reshape(shape)
        
        if viz_image is not None:
            heatmap = IImage(selected_vis, vmin=0, vmax=1).heatmap((selected_vis.shape[0]*up, selected_vis.shape[1]*up))
            video_frames_selfattn.append(viz_image+heatmap)
        else:
            heatmap = IImage(selected_vis, vmin=0).heatmap((selected_vis.shape[0]*up, selected_vis.shape[1]*up))
            video_frames_selfattn.append(heatmap)
    ###############################

    return out
    
