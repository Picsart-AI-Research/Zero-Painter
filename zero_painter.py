import argparse
from src.zeropainter.zero_painter_pipline import ZeroPainter
from src.zeropainter import models, dreamshaper,segmentation
from src.zeropainter import zero_painter_dataset

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask-path', default='data/masks/1_rgb.png',help='Mask path.')
    parser.add_argument('--metadata', default='data/metadata/1.json', type=str, help='Text prompt.')
    parser.add_argument('--output-dir', default='data/outputs/',help='Output dir.')
    
    #load models
    parser.add_argument('--config-folder-for-models', type=str, default='config', help='Path to configs')
    parser.add_argument('--model-folder-inpiting', type=str, default='models/sd-1-5-inpainting', help='Path to load inpainting model')
    parser.add_argument('--model-folder-generation', type=str, default='models/sd-1-4', help='Path to load generation model')
    parser.add_argument('--segment-anything-model', type=str, default='models/sam_vit_h_4b8939.pth', help='Path to load segmentation model')

    return parser.parse_args()


def main():
    args = get_args()

    model_inp,_ = models.get_inpainting_model(args.config_folder_for_models,args.model_folder_inpiting)
    model_t2i,_ = models.get_t2i_model(args.config_folder_for_models,args.model_folder_generation)
    model_sam = segmentation.get_segmentation_model(args.segment_anything_model)
    zero_painter_model = ZeroPainter(model_t2i, model_inp, model_sam)

    data = zero_painter_dataset.ZeroPainterDataset(args.mask_path, args.metadata)
    name = args.mask_path.split('/')[-1]
    result = zero_painter_model.gen_sample(data[0], 42, 42,30,30)
    result.save(args.output_dir+name)
    

if __name__ == '__main__':
    main()
