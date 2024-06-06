from . import segmentation, generation, inpainting

import torch
import numpy as np
from src.smplfusion import libimage
from src.smplfusion import IImage


class ZeroPainter:
    def __init__(self, model_t2i, model_inp, model_sam):
        self.model_t2i = model_t2i
        self.model_inp = model_inp
        self.model_sam = model_sam


    def gen_sample(
        self,
        sample,
        object_seed,
        image_seed,
        num_ddim_steps_t2i,
        num_ddim_steps_inp,
        cfg_scale_t2i=7.5,
        cfg_scale_inp=7.5,
        use_lcm_multistep_t2i=False,
        use_lcm_multistep_inp=False,
    ):
        

        gen_obj_list = []
        gen_mask_list = []
        real_mask_list = []

        if isinstance(object_seed, int):
            object_seed = [object_seed] * len(sample.masks)

        for i in range(len(sample.masks)):
            eps_list, z0_list, zt_list = generation.gen_single_object(
                self.model_t2i,
                sample.masks[i],
                object_seed[i],
                dt=1000 // num_ddim_steps_t2i,
                guidance_scale=cfg_scale_t2i,
                use_lcm_multistep=use_lcm_multistep_t2i,
            )
            mask = sample.masks[i].mask

            gen_image = IImage(
                self.model_t2i.vae.decode(z0_list[-1] / self.model_t2i.config.scale_factor)
            )
            gen_mask = segmentation.get_segmentation(self.model_sam, gen_image, mask)
            gen_object = gen_image * IImage(255 * gen_mask)

            gen_obj_list.append(gen_object)
            gen_mask_list.append(gen_mask)
            real_mask_list.append(mask)

        gen_image = IImage(libimage.stack(gen_obj_list).data.sum(0))
        gen_mask = IImage(255 * np.sum(gen_mask_list, 0))
        real_mask = IImage(255 * np.sum(real_mask_list, 0))

        output = inpainting.gen_filled_image(
            self.model_inp,
            sample.global_prompt,
            gen_image,
            (~gen_mask.alpha()).dilate(3),
            sample.masks,
            image_seed,
            dt=1000 // num_ddim_steps_inp,
            guidance_scale=cfg_scale_inp,
            use_lcm_multistep=use_lcm_multistep_inp,
        )

        return output



