import warnings 
warnings.filterwarnings('ignore')

import os
import json
import torch
import numpy as np
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import webcolors
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
from matplotlib.patches import Patch
plt.rcParams['axes.grid'] = False

from importlib import import_module
from argparse import ArgumentParser

from dataset import CustomDataLoader
from utils import label_accuracy_score, add_hist


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--anno_dir', type=str, default='/opt/ml/input/data/val.json') # 시각화할 데이터셋 anno
    parser.add_argument('--base_dir', type=str, default='/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-01/smp/trained_models/Unet_inceptionresnetv2_imagenet_221222_052351')
    parser.add_argument('--ckpt', type=str, default='latest.pt') # 사용할 pt 명
    parser.add_argument('--num_examples', type=int, default=16) # 확인할 이미지 개수
    parser.add_argument('--mode', type=str, default='eval') # 아직 사용 안해요, 추후 업데이트 예정입니다

    args = parser.parse_args()
    
    return args


def create_trash_label_colormap():
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    class_colormap = (
        ('Backgroud', 0, 0, 0),
        ('General trash', 192, 0, 128),
        ('Paper', 0, 128, 192),
        ('Paper pack', 0, 128, 64),
        ('Metal', 128, 0, 0),
        ('Glass', 64, 0, 128),
        ('Platic', 64, 0, 192),
        ('Styrofoam', 192, 128, 64),
        ('Plastic bag', 192, 192, 128),
        ('Battery', 64, 64, 128),
        ('Clothing', 128, 0, 192)
    )
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(class_colormap):
        colormap[inex] = [r, g, b]
    
    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint['net']


def plot_examples(args):
    """Visualization of images and masks according to batch size
    Args:
        mode: train/val/test (str)
        batch_id : 0 (int) 
        num_examples : 1 ~ batch_size(e.g. 8) (int)
        dataloaer : data_loader (dataloader) 
    Returns:
        None
    """
    category_names = (
        'Background',
        'General trash',
        'Paper',
        'Paper pack',
        'Metal',
        'Glass',
        'Plastic',
        'Styrofoam',
        'Plastic bag',
        'Battery',
        'Clothing'
    )

    class_colormap = [
        ['Backgroud',	    0,	0,	0,],
    	['General trash',	192,0,	128,],
    	['Paper',	        0,	128,192,],
    	['Paper pack',	    0,	128,64,],
    	['Metal',	        128,0,	0,],
    	['Glass',	        64,	0,	128,],
    	['Plastic',	        64,	0,	192,],
    	['Styrofoam',	    192,128,64,],
    	['Plastic bag',	    192,192,128,],
    	['Battery',	        64,	64,	128,],
    	['Clothing',	    128,0,	192,],
    ]

    config = json.load(open('/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-01/smp/trained_models/Unet_inceptionresnetv2_imagenet_221222_052351/config.json'))
    segmentation_model = config['segmentation_model']
    encoder_name = config['encoder_name']

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # variable for legend
    category_and_rgb = [[category, (r,g,b)] for idx, (category, r, g, b) in enumerate(class_colormap)]
    legend_elements = [Patch(facecolor=webcolors.rgb_to_hex(rgb), 
                             edgecolor=webcolors.rgb_to_hex(rgb), 
                             label=category) for category, rgb in category_and_rgb]

    
    transform = A.Compose([
                          ToTensorV2()
                          ])
    
    dataset = CustomDataLoader(data_dir=args.anno_dir, mode='train', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=4,
                                           collate_fn=collate_fn)

    model_module = getattr(import_module("segmentation_models_pytorch"), segmentation_model)
    model = model_module(
        encoder_name=encoder_name,
        in_channels=3,                        
        classes=11,                           
    )
    state_dict = load_model(model, os.path.join(args.base_dir, args.ckpt), device)
    model.load_state_dict(state_dict)

    
    # test / validation set에 대한 시각화
    n_class = 11
    model.eval()

    result = []
    with torch.no_grad():
        with tqdm(total=len(dataset)) as pbar:
            for index, (imgs, masks, image_infos) in enumerate(dataloader):
                hist = np.zeros((n_class, n_class))

                imgs = torch.stack(imgs)       
                masks = torch.stack(masks).long()

                imgs, masks = imgs.to(device), masks.to(device)            

                model = model.to(device)

                outs = model(imgs)

                imgs = imgs.detach().cpu().numpy()
                outs = torch.argmax(outs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()

                hist = add_hist(hist, masks, outs, n_class=n_class)
                
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
                    
                result.append([image_infos[0], imgs[0], masks[0], outs[0], mIoU, IoU_by_class])
                
                pbar.update(1)

        result.sort(key=lambda x:x[-2])
        print(f'IoU by class : {IoU_by_class}')

    # worst 저장
    fig, ax = plt.subplots(nrows=args.num_examples, ncols=3, figsize=(12, 4*args.num_examples), constrained_layout=True)
    for i, (image_infos, imgs, masks, outs, mIoU, IoU_by_class) in enumerate(result[:args.num_examples]):
        # Original Image
        ax[i][0].imshow(imgs.transpose([1,2,0]))
        ax[i][0].set_title(f"Orignal Image : {image_infos['file_name']}\n mIoU: {mIoU}")
        
        # Groud Truth
        ax[i][1].imshow(label_to_color_image(masks))
        ax[i][1].set_title(f"Groud Truth : {image_infos['file_name']}")
        
        # Pred Mask
        ax[i][2].imshow(label_to_color_image(outs))
        ax[i][2].set_title(f"Pred Mask : {image_infos['file_name']}")
        ax[i][2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    
    fig.tight_layout()
    plt.savefig(os.path.join(args.base_dir,args.ckpt.split('.')[0]+f'_worst_{args.num_examples}.jpg'))

    # best 저장
    fig, ax = plt.subplots(nrows=args.num_examples, ncols=3, figsize=(12, 4*args.num_examples), constrained_layout=True)
    for i, (image_infos, imgs, masks, outs, mIoU, IoU_by_class) in enumerate(result[-args.num_examples:]):
        # Original Image
        ax[i][0].imshow(imgs.transpose([1,2,0]))
        ax[i][0].set_title(f"Orignal Image : {image_infos['file_name']}\n mIoU: {mIoU}")
        
        # Groud Truth
        ax[i][1].imshow(label_to_color_image(masks))
        ax[i][1].set_title(f"Groud Truth : {image_infos['file_name']}")
        
        # Pred Mask
        ax[i][2].imshow(label_to_color_image(outs))
        ax[i][2].set_title(f"Pred Mask : {image_infos['file_name']}")
        ax[i][2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    
    fig.tight_layout()
    plt.savefig(os.path.join(args.base_dir,args.ckpt.split('.')[0]+f'_best_{args.num_examples}.jpg'))

def main():
    args = parse_args()
    plot_examples(args)

if __name__=='__main__':
    main()