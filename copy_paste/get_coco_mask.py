"""
get semantic segmentation annotations from coco data set.
"""
from PIL import Image
import numpy as np
import shutil
import imgviz
import argparse
import os
import tqdm
from pycocotools.coco import COCO

 
def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    # colormap = imgviz.label_colormap()
    colormap = np.array([
        [0, 0, 0], 
        [192, 0, 128], 
        [0, 128, 192], 
        [0, 128, 64], 
        [128, 0, 0], 
        [64, 0, 128], 
        [64, 0, 192], 
        [192, 128, 64], 
        [192, 192, 128], 
        [64, 64, 128], 
        [128, 0, 192]
        ], dtype = np.uint8)
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)
 
 
def main(args):
    annotation_file = os.path.join(args.input_dir, '{}.json'.format(args.split))
    os.makedirs(os.path.join(args.output_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'JPEGImages'), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
            img_origin_path = os.path.join(args.input_dir, img['file_name'])
            img_output_path = os.path.join(args.output_dir, 'JPEGImages', img['file_name'])
            seg_output_path = os.path.join(args.output_dir, 'SegmentationClass', img['file_name'].replace('.jpg', '.png'))
            if not os.path.exists(os.path.dirname(img_output_path)):
                os.makedirs(os.path.dirname(img_output_path))
            if not os.path.exists(os.path.dirname(seg_output_path)):
                os.makedirs(os.path.dirname(seg_output_path))
            shutil.copy(img_origin_path, img_output_path)
            save_colored_mask(mask, seg_output_path)
            # mask_pil = Image.fromarray(mask.astype(np.uint8))
            # mask_pil.save(seg_output_path)
 
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/opt/ml/input/data", type=str,
                        help="input directory")
    parser.add_argument("--output_dir", default="/opt/ml/input/data/copy_paste", type=str,
                        help="output directory")
    parser.add_argument("--split", default="train_all", type=str,
                        help="train or val")
    return parser.parse_args()
 
 
if __name__ == '__main__':
    args = get_args()
    main(args)