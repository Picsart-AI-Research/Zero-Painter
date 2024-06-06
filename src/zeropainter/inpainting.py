from src.smplfusion.common import *
import torch
from src.smplfusion import share
from src.smplfusion.patches import attentionpatch
from tqdm import tqdm 
from pytorch_lightning import seed_everything
from src.smplfusion.patches import router
from src.smplfusion import IImage

# negative_prompt = "worst quality, ugly, gross, disfigured, deformed, dehydrated, extra limbs, fused body parts, mutilated, malformed, mutated, bad anatomy, bad proportions, low quality, cropped, low resolution, out of frame, poorly drawn, text, watermark, letters, jpeg artifacts"
# positive_prompt = ", realistic, HD, Full HD, 4K, high quality, high resolution, masterpiece, trending on artstation, realistic lighting"
negative_prompt = ''
positive_prompt = '' 
VERBOSE = True

class AttnForward:
    def __init__(self, masks, object_context, object_uc_context):
        self.masks = masks
        self.object_context = object_context
        self.object_uc_context = object_uc_context

    def __call__(data, self, x, context=None, mask=None):
        att_type = "self" if context is None else "cross"
        batch_size = x.shape[0]

        if att_type == 'cross' and x.shape[1] in [share.input_shape.res16]: # For cross attention
            out = torch.zeros_like(x)
            for i in range(len(data.masks)):
                if data.masks[i].sum()>0:
                    if batch_size == 1:
                        out[:,data.masks[i]] = attentionpatch.default.forward(
                            self, 
                            x[:,data.masks[i] > 0], 
                            data.object_context[i][None]
                        ).float()
                    elif batch_size == 2:
                        out[:,data.masks[i]] = attentionpatch.default.forward(
                            self, 
                            x[:,data.masks[i] > 0], 
                            torch.stack([data.object_uc_context[i],data.object_context[i]])
                        ).float()
                    else:
                        raise NotImplementedError("Batch Size > 1 not yet supported!")
            return out
        else:
            return attentionpatch.default.forward(self, x, context, mask)

def gen_filled_image(model, prompt, image, mask, zp_masks, seed, T = 899, dt = 20, model_t2i = None, guidance_scale = 7.5, use_lcm_multistep = False):
    masks = [x.modified_mask.val16.flatten()>0 for x in zp_masks]
    masks.append(torch.stack(masks).sum(0) == 0)

    context = model.encoder.encode(['', f"realistic photo of a {prompt}" + positive_prompt])
    object_context = model.encoder.encode([x.local_prompt for x in zp_masks] + [f"realistic photo of a {prompt}" + positive_prompt])
    object_uc_context = model.encoder.encode([negative_prompt] * len(zp_masks) + [', '.join([x.local_prompt for x in zp_masks])])

    seed_everything(seed)

    eps = torch.randn((1,4,64,64)).cuda()
    # zT = schedule.sqrt_alphas[799] * condition_mask + schedule.sqrt_one_minus_alphas[799] * eps
    # zT = (condition_mask < 0) * zT + (condition_mask > 0) * eps
    # zT = schedule.sqrt_one_minus_alphas[899] * eps
    # zT = schedule.sqrt_alphas[899] * IImage(255*orig_masks_all).resize(64).alpha().torch().cuda() + schedule.sqrt_one_minus_alphas[899] * eps
    condition_x0 = model.vae.encode(image.torch().cuda() * ~mask.torch(0).bool().cuda()).mean * model.config.scale_factor
    condition_xT = model.schedule.sqrt_alphas[T] * condition_x0 + model.schedule.sqrt_one_minus_alphas[T] * eps
    condition_mask = mask.resize(64).cuda().torch(0).bool().float()
    zT = (condition_mask == 0) * condition_xT + (condition_mask > 0) * eps
    # zT = eps
    
    router.attention_forward = AttnForward(masks, object_context, object_uc_context)
    # router.attention_forward = attentionpatch.default.forward

    with torch.autocast('cuda'), torch.no_grad():        
        zt = zT
        timesteps = list(range(T, 0, -dt))
        pbar = tqdm(timesteps) if VERBOSE else timesteps
        for index,t in enumerate(pbar):
            if index == 0:
                current_mask = ~(~mask).dilate(2)
                condition_x0 = model.vae.encode(image.torch().cuda() * ~mask.cuda().torch(0).bool().cuda()).mean * model.config.scale_factor
                condition_mask = current_mask.resize(64).cuda().torch(0).bool().float()
                condition = torch.cat([condition_mask, condition_x0], 1)
            if index == 5:
                current_mask = ~(~mask).dilate(0)
                condition_x0 = model.vae.encode(image.torch().cuda() * ~mask.cuda().torch(0).bool().cuda()).mean * model.config.scale_factor
                condition_mask = current_mask.resize(64).cuda().torch(0).bool().float()
                condition = torch.cat([condition_mask, condition_x0], 1)
            if index == len(timesteps) - 5:
                if model_t2i is not None: 
                    model = model_t2i
                    condition = None
                else:
                    current_mask = mask.dilate(512)
                    condition_x0 = model.vae.encode(image.torch().cuda() * ~current_mask.cuda().torch(0).bool().cuda()).mean * model.config.scale_factor
                    condition_mask = current_mask.resize(64).cuda().torch(0).bool().float()
                    condition = torch.cat([condition_mask, condition_x0], 1)

            _zt = zt if condition is None else torch.cat([zt, condition], 1)

            if use_lcm_multistep:
                eps = model.unet(
                    _zt, 
                    timesteps = torch.tensor([t]).cuda(), 
                    context = context[1][None]
                )
            else:    
                eps_uncond, eps = model.unet(
                    torch.cat([_zt, _zt]), 
                    timesteps = torch.tensor([t, t]).cuda(), 
                    context = context
                ).chunk(2)
                eps = (eps_uncond + guidance_scale * (eps - eps_uncond))

            z0 = (zt - model.schedule.sqrt_one_minus_alphas[t] * eps) / model.schedule.sqrt_alphas[t]
            zt = model.schedule.sqrt_alphas[t - dt] * z0 + model.schedule.sqrt_one_minus_alphas[t - dt] * eps
            
    out = IImage(model.vae.decode(z0 / model.config.scale_factor))
    return out
    # # out.save('../../assets/paper_data/visuals_for_paper/'+str(seed_num)+'_'+name)

    # np_out = np.array(out.data[0])
    # print(np_out.max(), orig_masks_all.max())
    # blending_result_helper = 0.4*(np_out/255)+0.6*orig_masks_all
    # res = np.hstack([np_out,blending_result_helper*255])
    # n = name.split('.')[0]
    # cv2.imwrite('../../assets/paper_data/visuals_for_paper/'+n+'_'+str(my_seed_gen)+'__'+str(seed_num)+'_.png',res[:,:,::-1])
    # (IImage(255 * blending_result_helper) | IImage(np_out))