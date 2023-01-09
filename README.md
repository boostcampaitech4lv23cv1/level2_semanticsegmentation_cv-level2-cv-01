## Semantic Segmentation Competition
***
## Index
- [Semantic Segmentation Competition](#semantic-segmentation-competition)
- [Index](#index)
- [Members🔥](#members)
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
## Members🔥
| [김범준](https://github.com/quasar529) | [백우열](https://github.com/wooyeolBaek) | [조용재](https://github.com/yyongjae) | [조윤재](https://github.com/KidsareBornStars) | [최명헌](https://github.com/MyeongheonChoi) |
| :-: | :-: | :-: | :-: | :-: |
| <img src="https://avatars.githubusercontent.com/quasar529" width="100"> | <img src="https://avatars.githubusercontent.com/wooyeolBaek" width="100"> | <img src="https://avatars.githubusercontent.com/yyongjae" width="100"> | <img src="https://avatars.githubusercontent.com/KidsareBornStars" width="100"> | <img src="https://avatars.githubusercontent.com/MyeongheonChoi" width="100"> |
***
## Project Summary
<img src="images\description.png" width="100%" height="30%"/>

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나다. 
잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 
잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문이다.
따라서 사진에서 쓰레기를 Segmentation하는 모델을 만들어 이러한 문제점을 해결해보고자 한다. 문제 해결을 위한 데이터셋으로 11가지의 쓰레기가 찍힌 사진 데이터셋이 제공된다.

**Dataset**
<img src="images\dataset.png" width="100%" height="30%"/>
- 11개 Class
    - `Background`, `Generaltrash`, `Paper`, `Paperpack`, `Metal`, `Glass`, `Plastic`, `Styrofoam`, `Plastic bag`, `Battery`, `Clothing`
- 이미지 크기 : $512*512$
- Annotation file
    - COCO format : 크게 2가지(images, annotations)의 정보 포함.
    - images : `id`,`height`,`width`,`filename`
    - annotations : `id`,`segmentation`,`bbox`,`area`,`category_id`,`image_id`
***

## Procedures
<img src="images\Timeline.png" width="80%" height="50%"/>

**[2022.12.19 - 2022.12.21]**
- Semantic Segmentation 이론 학습.
- 사용할 라이브러리 결정 -> smp, mmsegmentation 사용법 숙지.

**[2022.12.21 - 2022.12.31]**
- Sweep으로 모델 체리피킹 (smp 사용)
  - Encoder, Decoder 조합 실험 결과
    - Encoder : PAN, FPN, DeepLavV3
    - Decoder : mit_b5, xception
      - DeepLabV3 계열의 경우 mit Decoder와 연동이 되지 않아 실험X.
  - Loss, Scheduler, Optimizer
    - Loss
      - CrossEntropy,DICE,Jaccard,Focal,Lovasz
      - Sweep 결과 Dice 와 CrossEntropy 사용.
    - Scheduler 
      - MultiStepLR,ExponentialLR,CosineAnnealingLR,CosineAnnealingWarmRestarts,OneCycleLR
      - Sweep 결과 CosineAnnealingLR,CosineAnnealingWarmRestarts 사용.
    - Optimizer
      - 실험 환경 : PAN + mit_b5 + CrossEntropyLoss
      - Adam : VAL 0.6987 / LB 0.6616
      - **AdamP : VAL 0.7015 / LB 0.6691**
      - AdamW : VAL 0.6968 / LB 0.6455
  - Augmentation 
    
    <a href='https://attractive-delivery-2c1.notion.site/f8a83a452dab4fed9d0de250ee6d6fd2?v=bed39e7a4ec3491baff52ffb92deddf7'>실험 페이지</a>
    
    <img src="/level2_semanticsegmentation_cv-level2-cv-01/images/aug1.png">
    <img src="/level2_semanticsegmentation_cv-level2-cv-01/images/aug2.png">
    


**[2023.01.01 - 2023.01.05]**
- Copy-Paste (https://github.com/qq995431104/Copy-Paste-for-Semantic-Segmentation)
  - inference 결과 가려진 물체에 대한 masking 결과가 좋지 않았음
  - 특정 class(General Trash, Plastic, Paper Pack)의 masking이 잘 되지 않았음
  - 이를 해결하기 위한 방법으로 Copy-Paste를 적용함
- CV strategy
  - annotation의 class와 area를 고려한 Stratified Group K-fold 사용

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
  
  # config에는 원하는 파일 경로를 넣어준다.
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
  
  # config에는 원하는 파일 경로를 넣어준다.
  ```
- Step 2: Inference
  ```
  inference.ipynb 이용
  ```
***
## Folder Structure
```
📂 level2_semanticsegmentation_cv-level2-cv-01
├── 📂 copy_paste
│      ├── 📂 src
│      │      └── 📑 create_annotations.py
│      ├── 📑 concatjson.ipynb
│      ├── 📑 copy_paste.py
│      ├── 📑 create-custom-coco-dataset.ipynb
│      └── 📑 get_coco_mask.ipynb
│      
├── 📂 images
│      ├── 📑 Augmentation_img-1.py
│      ├── 📑 Augmentation_img-2.py
│      ├── 📑 dataset.png
│      ├── 📑 description.png
│      └── 📑 Timeline.png
│            
├── 📂 mmsegmentation
│      ├── 📂 _trashsegmentation
│      │     ├── 📂 __base__
│      │     │    ├── 📂 datasets
│      │     │    │    └── 📑 upstage.py
│      │     │    │    
│      │     │    ├── 📂 models
│      │     │    │    ├── 📑 segformer_mit-b0.py
│      │     │    │    ├── 📑 segmenter_vit-b16_mask.py
│      │     │    │    ├── 📑 upernet_beit.py
│      │     │    │    ├── 📑 upernet_convnext.py
│      │     │    │    └── 📑 upernet_swin.py
│      │     │    │    
│      │     │    ├── 📂 schedules
│      │     │    │    ├── 📑 schedule_160k.py
│      │     │    │    └── 📑 schedule.py
│      │     │    │    
│      │     │    └── 📑 default_runtime.py
│      │     │    
│      │     ├── 📂 beit
│      │     │    ├── 📑 upernet_beit-base_8x2_640x640_160k_ade20k.py
│      │     │    ├── 📑 upernet_beit-base_640x640_160k_ade20k_ms.py
│      │     │    ├── 📑 upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py
│      │     │    └── 📑 upernet_beit-large_fp16_640x640_160k_ade20k_ms.py
│      │     │
│      │     ├── 📂 convnext
│      │     │    ├── 📑 upernet_convnext_base_fp16_512x512_160k_ade20k.py
│      │     │    └── 📑 upernet_convnext_xlarge_fp16_640x640_160k_ade20k.py
│      │     │
│      │     ├── 📂 segformer
│      │     │    ├── 📑 segformer_mit-b0_512x512_160k_ade20k.py
│      │     │    └── 📑 segformer_mit-b5_512x512_160k_ade20k.py
│      │     │
│      │     ├── 📂 segmenter
│      │     │    └── 📑 segmenter_vit-l_mask_8x1_640x640_160k_ade20k.py
│      │     │
│      │     ├── 📂 swin
│      │     │    ├── 📑 upernet_swin_large_patch4_window7_512x512_pretrain_224x224_22K_160k_ade20k.py
│      │     │    ├── 📑 upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.py
│      │     │    └── 📑 upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
│      │     │
│      │     └── 📂 utils
│      │          ├── 📑 convert2mmseg.py
│      │          ├── 📑 inference.ipynb
│      │          └── 📑 inference.py
│      │
│      ├── 📂 config
│      │
│      ├── 📂 mmseg
│      │
│      ├── 📂 requirements
│      │
│      ├── 📂 submission
│      │
│      ├── 📂 tests
│      │
│      └── 📂 tools
│           ├── 📑 train.py
│           ├── 📑 test.py
│           ├── 📂 convert_datasets
│           ├── 📂 model_converters
│           └── ...
│
├── 📂 smp
│     ├── 📂 submission
│     ├── 📑 dataset.py
│     ├── 📑 eval.py
│     ├── 📑 inference.py
│     ├── 📑 requirements.txt
│     ├── 📑 train.py
│     └── 📑 utils.py
│
│
│── 📂 ViT-Adapter
│     ├── 📂 segmentation
│     │     ├── 📂 configs
│     │     │    ├── 📂 _base_
│     │     │    │    ├── 📂 datasets
│     │     │    │    │    └── 📑 upstage.py
│     │     │    │    │
│     │     │    │    ├── 📂 models
│     │     │    │    │    ├── 📑 mask2former_beit_upstage.py
│     │     │    │    │    └── 📑 upernet_beit_upstage.py
│     │     │    │    │
│     │     │    │    ├── 📂 schedules
│     │     │    │    │    ├── 📑 schedule.py
│     │     │    │    │    └── 📑 schedule_fp16.py
│     │     │    │    │    
│     │     │    │    └── 📑 default_runtime.py
│     │     │    │
│     │     │    └── 📂 upstage
│     │     │         ├── 📑 mask2former_beit_adapter_base_512_upstage_ss.py
│     │     │         ├── 📑 mask2former_beit_adapter_large_512_upstage_ss.py
│     │     │         ├── 📑 upernet_augreg_adapter_base_512_160k_upstage.py
│     │     │         └── 📑 upernet_deit_adapter_tiny_512_160k_upstage.py
│     │     │
│     │     ├── 📂 mmcv_custom
│     │     ├── 📂 mmseg_custom
│     │     ├── 📑 inference.ipynb
│     │     ├── 📑 train.py
│     │     ├── 📑 train_fp16.py
│     │     └── ...
│     └── 📑 ...
│
│
└── 📂 utils
│     ├── 📑 5fold.ipynb
│     ├── 📑 cleansing_by_feat.py
│     ├── 📑 cleansing_by_name.py
│     ├── 📑 combine_coco.py
│     ├── 📑 eda.py
│     ├── 📑 post_processing.py
│     └── 📑 size.png
│
└── 📑 README.md
```