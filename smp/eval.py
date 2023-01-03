import warnings 
warnings.filterwarnings('ignore')

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import albumentations as A
from albumentations.pytorch import ToTensorV2

import webcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['axes.grid'] = False

from importlib import import_module
from argparse import ArgumentParser

from dataset import CustomDataLoader
from utils import label_accuracy_score, add_hist, create_trash_label_colormap, label_to_color_image


def parse_args():
    parser = ArgumentParser()
    
    # 위에 세개만 지정해주고 실행하면 됨
    parser.add_argument('--anno_dir', type=str, default='/opt/ml/input/data/val.json') # 시각화할 데이터셋 anno(train, val, test중 하나는 반드시 json 이름에 포함돼 있어야함)
    parser.add_argument('--base_dir', type=str, default='/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-01/smp/trained_models/FPN_mit_b4_AdamW_221225_091349')
    parser.add_argument('--ckpt', type=str, default='latest.pt') # 사용할 pt 명
    
    parser.add_argument('--num_examples', type=int, default=16) # 확인할 이미지 개수
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data') # 이미지 불러올 때 사용
    parser.add_argument('--mode', type=str, default='train') # 따로 설정할 필요 없습니다(anno_dir에서 가져와요)

    args = parser.parse_args()

    args.mode = 'test' if 'test' in args.anno_dir else ('train' if 'train' in args.anno_dir else 'val')
    
    return args

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint['net']

def sort_save(num_examples, result, base_dir, ckpt, legend_elements, add_name, data_dir):
    add_name.replace(' ','_')
    fig, ax = plt.subplots(nrows=num_examples, ncols=4, figsize=(16, 4*num_examples), constrained_layout=True)
    for i, (image_infos, imgs, masks, outs, score) in enumerate(result):
        # Original Image
        ax[i][0].imshow(plt.imread(os.path.join(data_dir, image_infos)))
        ax[i][0].set_title(f"Orignal Image : {image_infos}\n {add_name if 'mIoU' in add_name else add_name+'_IoU'}: {score}")

        # transformed Image
        ax[i][1].imshow(imgs.transpose([1,2,0]).astype('uint8'))
        ax[i][1].set_title(f"Transformed Image : {image_infos}")
            
        # Groud Truth
        ax[i][2].imshow(label_to_color_image(masks).astype('uint8'))
        ax[i][2].set_title(f"Groud Truth : {image_infos}")
            
        # Pred Mask
        ax[i][3].imshow(label_to_color_image(outs).astype('uint8'))
        ax[i][3].set_title(f"Pred Mask : {image_infos}")
        ax[i][3].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        
    fig.tight_layout()
    plt.savefig(os.path.join(base_dir,ckpt.split('.')[0]+f'_{num_examples}_{add_name}.jpg'))

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

    config = json.load(open(os.path.join(args.base_dir,'config.json')))
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
                        A.Normalize(
                                mean=[0.46009142, 0.43957697, 0.41827273],
                                std=[0.21060736, 0.20755924, 0.21633709],
                                max_pixel_value=1.0,
                            ),
                        ToTensorV2()
                        ])
    
    dataset = CustomDataLoader(data_dir=args.anno_dir, mode=args.mode, transform=transform)
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
    
    result_by_class = defaultdict(list) # result_by_class['Paper'] = [[image_name 1, mIoU for paper], [image_name 2, mIoU for paper], ...], mi

    if args.mode != 'test':

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

                    # 이미지별 각 클래스의 IoU 추가
                    for name, iou in zip(category_names, IoU):
                        if np.isnan(iou):
                            continue
                        result_by_class[name].append([image_infos[0]['file_name'], imgs[0], masks[0], outs[0], iou])

                    # 이미지별 mIoU 저장
                    result_by_class['mIoU'].append([image_infos[0]['file_name'], imgs[0], masks[0], outs[0], mIoU])
                    
                    pbar.update(1)

        for k in result_by_class.keys():
            result_by_class[k].sort(key=lambda x:x[-1])

        # mIoU 이미지 worst & best 저장
        sort_save(args.num_examples, result_by_class['mIoU'][:args.num_examples], args.base_dir, args.ckpt, legend_elements, 'mIoU_worst', args.data_dir)
        sort_save(args.num_examples, result_by_class['mIoU'][-args.num_examples:], args.base_dir, args.ckpt, legend_elements, 'mIoU_best', args.data_dir)
        print('mIoU images are saved')

        # 클래스별 IoU 이미지 저장
        for k in category_names:
            sort_save(args.num_examples, result_by_class[k][:args.num_examples], args.base_dir, args.ckpt, legend_elements, k, args.data_dir)
            print(f'a {k} image is saved')

    else:
        result = []
        with torch.no_grad():
            with tqdm(total=len(dataset)) as pbar:
                for index, (imgs, image_infos) in enumerate(dataloader):

                    imgs = torch.stack(imgs)       

                    imgs = imgs.to(device)

                    model = model.to(device)

                    outs = model(imgs)

                    imgs = imgs.detach().cpu().numpy()
                    outs = torch.argmax(outs, dim=1).detach().cpu().numpy()
                        
                    result.append([image_infos[0], imgs[0], outs[0]])
                    
                    pbar.update(1)

        # preds 저장
        fig, ax = plt.subplots(nrows=args.num_examples, ncols=2, figsize=(12, 4*args.num_examples), constrained_layout=True)
        for i, (image_infos, imgs, outs) in enumerate(result[:args.num_examples]):
            # Original Image
            ax[i][0].imshow(plt.imread(os.path.join(args.data_dir, image_infos)))
            ax[i][0].set_title(f"Orignal Image : {image_infos}")
            
            # Transformed Image
            ax[i][1].imshow(imgs.transpose([1,2,0]))
            ax[i][1].set_title(f"Orignal Image : {image_infos['file_name']}")
            
            # Pred Mask
            ax[i][2].imshow(label_to_color_image(outs))
            ax[i][2].set_title(f"Pred Mask : {image_infos['file_name']}")
            ax[i][2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        
        fig.tight_layout()
        plt.savefig(os.path.join(args.base_dir,args.ckpt.split('.')[0]+f'_test_{args.num_examples}.jpg'))



def main():
    args = parse_args()
    plot_examples(args)

if __name__=='__main__':
    main()