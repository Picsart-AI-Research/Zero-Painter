import numpy as np
from os.path import dirname


from segment_anything import SamPredictor, sam_model_registry

def get_bounding_box(mask):
    all_y,all_x = np.where(mask == 1)
    all_y,all_x = sorted(all_y),sorted(all_x)
    x1,y1,x2,y2 = all_x[0],all_y[0],all_x[-1],all_y[-1]
    x,y,w,h = x1,y1,x2-x1,y2-y1
    return x,y,w,h

def get_segmentation_model(path_to_check):
    # path_to_check = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=path_to_check)
    predictor = SamPredictor(sam)
    predictor.model.cuda();
    return predictor

def get_segmentation(predictor, im, mask):
    im_np = im.data[0]

    # SAM prediction
    # mask = obj['data_512'][i]['mask']
    x,y,w,h = get_bounding_box(mask)
    predictor.set_image(im_np)
    input_box = np.array([x,y,x+w,y+h])
    masks, scores, other = predictor.predict(box=input_box)

    _masks = np.concatenate([masks, ~masks])
    _scores = np.concatenate([scores, scores])
    _scores = np.array([y for x,y in zip(_masks, _scores) if ((1 - mask) * x).sum() <= (mask * x).sum()])
    _masks = np.array([x for x in _masks if ((1 - mask) * x).sum() <= (mask * x).sum()])
    if len(_masks) > 0:
        masks,scores = _masks,_scores
    
    pred_seg_mask = masks[scores.argmax()]
    pred_seg_mask[pred_seg_mask > 0] = 1
    pred_seg_mask = np.stack([pred_seg_mask] * 3, axis=-1) * 1

    return pred_seg_mask * mask[...,None]
