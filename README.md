## Semantic Segmentation Competition
***
## Index
- [Semantic Segmentation Competition](#semantic-segmentation-competition)
- [Index](#index)
- [Members๐ฅ](#members)
- [Project Summary](#project-summary)
- [Procedures](#procedures)
- [Result](#result)
- [How to Run](#how-to-run)
  - [Requirements](#requirements)
  - [MMSegmentation](#mmsegmentation)
  - [SMP](#smp)
  - [ViT-Adapter](#vit-adapter)
- [Folder Structure](#folder-structure)
***
## Members๐ฅ
| [๊น๋ฒ์ค](https://github.com/quasar529) | [๋ฐฑ์ฐ์ด](https://github.com/wooyeolBaek) | [์กฐ์ฉ์ฌ](https://github.com/yyongjae) | [์กฐ์ค์ฌ](https://github.com/KidsareBornStars) | [์ต๋ชํ](https://github.com/MyeongheonChoi) |
| :-: | :-: | :-: | :-: | :-: |
| <img src="https://avatars.githubusercontent.com/quasar529" width="100"> | <img src="https://avatars.githubusercontent.com/wooyeolBaek" width="100"> | <img src="https://avatars.githubusercontent.com/yyongjae" width="100"> | <img src="https://avatars.githubusercontent.com/KidsareBornStars" width="100"> | <img src="https://avatars.githubusercontent.com/MyeongheonChoi" width="100"> |
***
## Project Summary
<img src="images\description.png" width="100%" height="30%"/>

๋ถ๋ฆฌ์๊ฑฐ๋ ์ด๋ฌํ ํ๊ฒฝ ๋ถ๋ด์ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ ์ค ํ๋๋ค. 
์ ๋ถ๋ฆฌ๋ฐฐ์ถ ๋ ์ฐ๋ ๊ธฐ๋ ์์์ผ๋ก์ ๊ฐ์น๋ฅผ ์ธ์ ๋ฐ์ ์ฌํ์ฉ๋์ง๋ง, 
์๋ชป ๋ถ๋ฆฌ๋ฐฐ์ถ ๋๋ฉด ๊ทธ๋๋ก ํ๊ธฐ๋ฌผ๋ก ๋ถ๋ฅ๋์ด ๋งค๋ฆฝ ๋๋ ์๊ฐ๋๊ธฐ ๋๋ฌธ์ด๋ค.
๋ฐ๋ผ์ ์ฌ์ง์์ ์ฐ๋ ๊ธฐ๋ฅผ Segmentationํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ์ด๋ฌํ ๋ฌธ์ ์ ์ ํด๊ฒฐํด๋ณด๊ณ ์ ํ๋ค. ๋ฌธ์  ํด๊ฒฐ์ ์ํ ๋ฐ์ดํฐ์์ผ๋ก 11๊ฐ์ง์ ์ฐ๋ ๊ธฐ๊ฐ ์ฐํ ์ฌ์ง ๋ฐ์ดํฐ์์ด ์ ๊ณต๋๋ค.

**Dataset**
<img src="images\dataset.png" width="100%" height="30%"/>
- 11๊ฐ Class
    - `Background`,ย `Generaltrash`,ย `Paper`,ย `Paperpack`,ย `Metal`,ย `Glass`,ย `Plastic`,ย `Styrofoam`,ย `Plastic bag`,ย `Battery`,ย `Clothing`
- ์ด๋ฏธ์ง ํฌ๊ธฐ : $512*512$
- Annotation file
    - COCO format : ํฌ๊ฒ 2๊ฐ์ง(images, annotations)์ ์ ๋ณด ํฌํจ.
    - images : `id`,`height`,`width`,`filename`
    - annotations : `id`,`segmentation`,`bbox`,`area`,`category_id`,`image_id`
***

## Procedures
<img src="images\Timeline.png" width="80%" height="50%"/>

**[2022.12.19 - 2022.12.21]**
- Semantic Segmentation ์ด๋ก  ํ์ต.
- ์ฌ์ฉํ  ๋ผ์ด๋ธ๋ฌ๋ฆฌ ๊ฒฐ์  -> smp, mmsegmentation ์ฌ์ฉ๋ฒ ์์ง.

**[2022.12.21 - 2022.12.31]**
- Sweep์ผ๋ก ๋ชจ๋ธ ์ฒด๋ฆฌํผํน (smp ์ฌ์ฉ)
  - Encoder, Decoder ์กฐํฉ ์คํ ๊ฒฐ๊ณผ
    - Encoder : PAN, FPN, DeepLavV3
    - Decoder : mit_b5, xception
      - DeepLabV3 ๊ณ์ด์ ๊ฒฝ์ฐ mit Decoder์ ์ฐ๋์ด ๋์ง ์์ ์คํX.
  - Loss, Scheduler, Optimizer
    - Loss
      - CrossEntropy,DICE,Jaccard,Focal,Lovasz
      - Sweep ๊ฒฐ๊ณผ Dice ์ CrossEntropy ์ฌ์ฉ.
    - Scheduler 
      - MultiStepLR,ExponentialLR,CosineAnnealingLR,CosineAnnealingWarmRestarts,OneCycleLR
      - Sweep ๊ฒฐ๊ณผ CosineAnnealingLR,CosineAnnealingWarmRestarts ์ฌ์ฉ.
    - Optimizer
      - ์คํ ํ๊ฒฝ : PAN + mit_b5 + CrossEntropyLoss
      - Adam : VAL 0.6987 / LB 0.6616
      - **AdamP : VAL 0.7015 / LB 0.6691**
      - AdamW : VAL 0.6968 / LB 0.6455
  - Augmentation 
    
    <a href='https://attractive-delivery-2c1.notion.site/f8a83a452dab4fed9d0de250ee6d6fd2?v=bed39e7a4ec3491baff52ffb92deddf7'>์คํ ํ์ด์ง</a>
    
    <img src="images/aug1.png" />
    <img src="images/aug2.png" />
    


**[2023.01.01 - 2023.01.05]**
- Copy-Paste (https://github.com/qq995431104/Copy-Paste-for-Semantic-Segmentation)
  - inference ๊ฒฐ๊ณผ ๊ฐ๋ ค์ง ๋ฌผ์ฒด์ ๋ํ masking ๊ฒฐ๊ณผ๊ฐ ์ข์ง ์์์
  - ํน์  class(General Trash, Plastic, Paper Pack)์ masking์ด ์ ๋์ง ์์์
  - ์ด๋ฅผ ํด๊ฒฐํ๊ธฐ ์ํ ๋ฐฉ๋ฒ์ผ๋ก Copy-Paste๋ฅผ ์ ์ฉํจ
- CV strategy
  - annotation์ class์ area๋ฅผ ๊ณ ๋ คํ Stratified Group K-fold ์ฌ์ฉ

**[2023.01.02 - 2023.01.05]**
- Pseudo Labeling
- Fold Ensemble
- Post-processing
***
## Result
|  | mIoU | Ranking |
| --- | --- | --- |
| PUBLIC | 0.7665 | 9th |
| PRIVATE | 0.7563 | 8th |
***
## How to Run

### Requirements

- python==3.8.2
- pytorch==1.7.1
- opencv-python==4.5.5.64
- Pillow==9.1.0
- pandas==1.3.4
- pycocotools==2.0.4
- seaborn==0.11.2
- segmentation-models-pytorch==0.2.0
- scipy==1.7.3
- webcolors==1.11.1
- albumentations==1.0.3
- mmsegmentation==0.29.1
- mmcv-full==1.4.2
- torchvision==0.10.0
- torchaudio==0.9.0
<br>

### MMSegmentation

- Step 1: Train
  ```
  ex)
  
  python mmsegmentation/tools/train.py \ _trashsegmentation/convnext/upernet_convnext_xlarge_fp16_640x640_160k_ade20k.py
  
  # config์๋ ์ํ๋ ํ์ผ ๊ฒฝ๋ก๋ฅผ ๋ฃ์ด์ค๋ค.
  ```
- Step 2: Inference
  ```
  python mmsegmentation/_trashsegmentation/utils/inference.py
  ```

### SMP

- Step 1: Train
  ```
  python smp/train.py
  ```
- Step 2: Inference
  ```
  python smp/inference.py
  ```

### ViT-Adapter

- Step 0: Change Directory
  ```  
  cd ViT-Adapter/segmentation
  ```
- Step 1: Train
  ```
  ex)
  
  python train.py configs/upstage/mask2former_beit_adapter_base_512_upstage_ss.py
  
  # config์๋ ์ํ๋ ํ์ผ ๊ฒฝ๋ก๋ฅผ ๋ฃ์ด์ค๋ค.
  ```
- Step 2: Inference
  ```
  inference.ipynb ์ด์ฉ
  ```
***
## Folder Structure
```
๐ level2_semanticsegmentation_cv-level2-cv-01
โโโ ๐ copy_paste
โ      โโโ ๐ src
โ      โ      โโโ ๐ create_annotations.py
โ      โโโ ๐ concatjson.ipynb
โ      โโโ ๐ copy_paste.py
โ      โโโ ๐ create-custom-coco-dataset.ipynb
โ      โโโ ๐ get_coco_mask.ipynb
โ      
โโโ ๐ images
โ      โโโ ๐ Augmentation_img-1.py
โ      โโโ ๐ Augmentation_img-2.py
โ      โโโ ๐ dataset.png
โ      โโโ ๐ description.png
โ      โโโ ๐ Timeline.png
โ            
โโโ ๐ mmsegmentation
โ      โโโ ๐ _trashsegmentation
โ      โ     โโโ ๐ __base__
โ      โ     โ    โโโ ๐ datasets
โ      โ     โ    โ    โโโ ๐ upstage.py
โ      โ     โ    โ    
โ      โ     โ    โโโ ๐ models
โ      โ     โ    โ    โโโ ๐ segformer_mit-b0.py
โ      โ     โ    โ    โโโ ๐ segmenter_vit-b16_mask.py
โ      โ     โ    โ    โโโ ๐ upernet_beit.py
โ      โ     โ    โ    โโโ ๐ upernet_convnext.py
โ      โ     โ    โ    โโโ ๐ upernet_swin.py
โ      โ     โ    โ    
โ      โ     โ    โโโ ๐ schedules
โ      โ     โ    โ    โโโ ๐ schedule_160k.py
โ      โ     โ    โ    โโโ ๐ schedule.py
โ      โ     โ    โ    
โ      โ     โ    โโโ ๐ default_runtime.py
โ      โ     โ    
โ      โ     โโโ ๐ beit
โ      โ     โ    โโโ ๐ upernet_beit-base_8x2_640x640_160k_ade20k.py
โ      โ     โ    โโโ ๐ upernet_beit-base_640x640_160k_ade20k_ms.py
โ      โ     โ    โโโ ๐ upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py
โ      โ     โ    โโโ ๐ upernet_beit-large_fp16_640x640_160k_ade20k_ms.py
โ      โ     โ
โ      โ     โโโ ๐ convnext
โ      โ     โ    โโโ ๐ upernet_convnext_base_fp16_512x512_160k_ade20k.py
โ      โ     โ    โโโ ๐ upernet_convnext_xlarge_fp16_640x640_160k_ade20k.py
โ      โ     โ
โ      โ     โโโ ๐ segformer
โ      โ     โ    โโโ ๐ segformer_mit-b0_512x512_160k_ade20k.py
โ      โ     โ    โโโ ๐ segformer_mit-b5_512x512_160k_ade20k.py
โ      โ     โ
โ      โ     โโโ ๐ segmenter
โ      โ     โ    โโโ ๐ segmenter_vit-l_mask_8x1_640x640_160k_ade20k.py
โ      โ     โ
โ      โ     โโโ ๐ swin
โ      โ     โ    โโโ ๐ upernet_swin_large_patch4_window7_512x512_pretrain_224x224_22K_160k_ade20k.py
โ      โ     โ    โโโ ๐ upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.py
โ      โ     โ    โโโ ๐ upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
โ      โ     โ
โ      โ     โโโ ๐ utils
โ      โ          โโโ ๐ convert2mmseg.py
โ      โ          โโโ ๐ inference.ipynb
โ      โ          โโโ ๐ inference.py
โ      โ
โ      โโโ ๐ config
โ      โ
โ      โโโ ๐ mmseg
โ      โ
โ      โโโ ๐ requirements
โ      โ
โ      โโโ ๐ submission
โ      โ
โ      โโโ ๐ tests
โ      โ
โ      โโโ ๐ tools
โ           โโโ ๐ train.py
โ           โโโ ๐ test.py
โ           โโโ ๐ convert_datasets
โ           โโโ ๐ model_converters
โ           โโโ ...
โ
โโโ ๐ smp
โ     โโโ ๐ submission
โ     โโโ ๐ dataset.py
โ     โโโ ๐ eval.py
โ     โโโ ๐ inference.py
โ     โโโ ๐ requirements.txt
โ     โโโ ๐ train.py
โ     โโโ ๐ utils.py
โ
โ
โโโ ๐ ViT-Adapter
โ     โโโ ๐ segmentation
โ     โ     โโโ ๐ configs
โ     โ     โ    โโโ ๐ _base_
โ     โ     โ    โ    โโโ ๐ datasets
โ     โ     โ    โ    โ    โโโ ๐ upstage.py
โ     โ     โ    โ    โ
โ     โ     โ    โ    โโโ ๐ models
โ     โ     โ    โ    โ    โโโ ๐ mask2former_beit_upstage.py
โ     โ     โ    โ    โ    โโโ ๐ upernet_beit_upstage.py
โ     โ     โ    โ    โ
โ     โ     โ    โ    โโโ ๐ schedules
โ     โ     โ    โ    โ    โโโ ๐ schedule.py
โ     โ     โ    โ    โ    โโโ ๐ schedule_fp16.py
โ     โ     โ    โ    โ    
โ     โ     โ    โ    โโโ ๐ default_runtime.py
โ     โ     โ    โ
โ     โ     โ    โโโ ๐ upstage
โ     โ     โ         โโโ ๐ mask2former_beit_adapter_base_512_upstage_ss.py
โ     โ     โ         โโโ ๐ mask2former_beit_adapter_large_512_upstage_ss.py
โ     โ     โ         โโโ ๐ upernet_augreg_adapter_base_512_160k_upstage.py
โ     โ     โ         โโโ ๐ upernet_deit_adapter_tiny_512_160k_upstage.py
โ     โ     โ
โ     โ     โโโ ๐ mmcv_custom
โ     โ     โโโ ๐ mmseg_custom
โ     โ     โโโ ๐ inference.ipynb
โ     โ     โโโ ๐ train.py
โ     โ     โโโ ๐ train_fp16.py
โ     โ     โโโ ...
โ     โโโ ๐ ...
โ
โ
โโโ ๐ utils
โ     โโโ ๐ 5fold.ipynb
โ     โโโ ๐ cleansing_by_feat.py
โ     โโโ ๐ cleansing_by_name.py
โ     โโโ ๐ combine_coco.py
โ     โโโ ๐ eda.py
โ     โโโ ๐ post_processing.py
โ     โโโ ๐ size.png
โ
โโโ ๐ README.md
```
