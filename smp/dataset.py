import os
import cv2
import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def Mixup(images, masks, alpha=0.2):
    # masks: one-hot encoding된 y
    x1, y1 = images[0], masks[0]
    x2, y2 = images[1], masks[1]

    lambda_param = np.random.beta(alpha, alpha)
    images = lambda_param * x1 + (1 - lambda_param) * x2
    masks = lambda_param * y1 + (1 - lambda_param) * y2

    return images, masks

def get_transform(mode='train', preprocessing_fn=None):
    transform = []

    if mode=='train':
        transform = [
            # geometric
            #A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height = 384, width = 384, ignore_values=[[0,0,0]]),
            A.RandomResizedCrop(512, 512, (0.1, 1.0), p=0.5),
            A.GridDropout(ratio=0.2, random_offset=True, holes_number_x=4, holes_number_y=4, p=0.1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.ShiftScaleRotate(p=0.2, shift_limit=0.2, scale_limit=0.2, rotate_limit=45),
                A.RandomRotate90(),
            ], p=0.2),
            #A.RandomRotate90(),

            # style
            A.OneOf([
                A.ChannelShuffle(),
                A.ToGray(),
            ],p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            
        ]
    
    transform.append(A.Lambda(image=preprocessing_fn))
    transform.append(A.Normalize(
                mean=[0.46009142, 0.43957697, 0.41827273],
                std=[0.21060736, 0.20755924, 0.21633709],
                max_pixel_value=1.0,
            ))
    transform.append(ToTensorV2())

    return A.Compose(transform)
    

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.categories = [
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
            ]
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        dataset_path  = '/opt/ml/input/data'
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            
            category_names = list(self.categories)

            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
