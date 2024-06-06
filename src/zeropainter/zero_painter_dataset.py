import json
import torch
import cv2
import numpy as np
from src.smplfusion import share

class ZeroPainterMask:
    def __init__(self, color, local_prompt, img_grey,rgb,image_w,image_h):
        self.img_grey = img_grey
        self.color = eval(color)
        self.local_prompt = local_prompt
        self.bbox = None
        self.bbox_64 = None
        self.area = None
        self.mask = None
        self.mask_64 = None
        self.token_idx = None
        self.modified_mask = None
        self.inverse_mask = None
        self.modified_indexses_of_prompt = None
        self.sot_index = [0]
        self.w_positive = 1
        self.w_negative = 1
        self.image_w = image_w
        self.image_h = image_h
        self.rgb = rgb
        self.modify_info()

    def get_bounding_box(self, mask):
        mask = mask*255.0
        all_y, all_x = np.where(mask == 255.0)
        all_y, all_x = sorted(all_y), sorted(all_x)
        x1, y1, x2, y2 = all_x[0], all_y[0], all_x[-1], all_y[-1]
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        return x, y, w, h
    

    def modify_info(self):
        # Define the color to be searched for
        if self.rgb:
            color_to_search = np.array([self.color[0],self.color[1],self.color[2]])
            mask_1d = np.all(self.img_grey == color_to_search, axis=2)*1.0
        else:
            mask_1d = (self.img_grey == self.color) * 1.0
        mask_1d_64 = cv2.resize(mask_1d, (64, 64), interpolation=cv2.INTER_NEAREST)
        self.bbox = self.get_bounding_box(mask_1d)
        self.bbox_64 = self.get_bounding_box(mask_1d_64)
        self.area = self.bbox[2] * self.bbox[3]
        self.mask = mask_1d.copy()
        self.mask_64 = mask_1d_64 // 255

        splited_prompt = self.local_prompt.split()
        self.token_idx = np.arange(1, len(splited_prompt) + 1, 1, dtype=int)

        #dif part
        mask_1d = mask_1d[None,None,:,:]
        mask_1d = torch.from_numpy(mask_1d)

        self.modified_indexses_of_prompt = np.arange(5, len(splited_prompt)+1)
        self.modified_mask = share.InputMask(mask_1d)
        self.inverse_mask = share.InputMask(1-mask_1d)
        
        return

class ZeroPainterSample:
    #Example of sample
    #{'prompt': 'Brown gift box beside red candle.',
    # 'color_context_dict': {'1': 'Brown gift box', '2': 'red candle'}}
    def __init__(self, item, img_grey, image_h=512,image_w=512):
        
        self.item = item
        self.global_prompt = self.item['prompt']
        self.image_w = image_w
        self.image_h = image_h
        self.img_grey = img_grey
        self.masks = self.load_masks()
       
          
    def load_masks(self):
        data_samples = []
        
        self.img_grey = cv2.resize(self.img_grey, (self.image_w, self.image_h),interpolation=cv2.INTER_NEAREST)
        for color, local_prompt in self.item['color_context_dict'].items():
            data_samples.append(ZeroPainterMask(color,local_prompt,self.img_grey,self.item['rgb'],self.image_w,self.image_h))

        return data_samples

    
class ZeroPainterDataset:
    def __init__(self, root_path_img, json_path,rgb=True):
        self.root_path_img = root_path_img
        self.json_path = json_path
        self.rgb = rgb

        if isinstance(json_path, dict) or isinstance(json_path, list):
            self.json_data = json_path
        else:
            with open(self.json_path, 'r') as file:
                self.json_data = json.load(file)
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        item = self.json_data[index]
        item['rgb'] = self.rgb    
        if isinstance(self.root_path_img, str):
            self.img_path = self.root_path_img
            if self.rgb:
                self.img_grey = cv2.imread(self.img_path)  
            else:
                self.img_grey = cv2.imread(self.img_path,0)      
        else:
            self.img_grey = np.array(self.root_path_img)
            
        return ZeroPainterSample(item, self.img_grey)


