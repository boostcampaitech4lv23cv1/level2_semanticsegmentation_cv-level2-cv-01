import os
import math
import json
import random
import warnings 
warnings.filterwarnings('ignore')

import torch
from utils import label_accuracy_score, add_hist
from importlib import import_module

import numpy as np
from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR, MultiStepLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

from argparse import ArgumentParser

from pytz import timezone
from datetime import datetime
import wandb

import segmentation_models_pytorch as smp

from dataset import *

# when using xception encoder
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def parse_args():
    parser = ArgumentParser()

    # model
    parser.add_argument('--segmentation_model', type=str, default='Unet')       # [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]
    parser.add_argument('--encoder_name', type=str, default='resnet34')
    parser.add_argument('--encoder_weights', type=str, default='imagenet')

    # path
    parser.add_argument('--saved_dir', type=str, default='trained_models')

    # dataset path
    parser.add_argument('--train_path', type=str, default='/opt/ml/input/data/train.json')
    parser.add_argument('--valid_path', type=str, default='/opt/ml/input/data/val.json')

    # hyperparameters
    parser.add_argument('--num_epochs', type=int, default=40) # 20
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss')    # [CrossEntropyLoss, JaccardLoss, DiceLoss, FocalLoss, LovaszLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss, TverskyLoss, MCCLoss]
    parser.add_argument('--learning_rate', type=float, default=1e-4) # 1e-4
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--num_workers', type=int, default=4)

    # mixup
    parser.add_argument('--mixup', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.2)
    
    # early stopping
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=7)

    # settings
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # wandb
    parser.add_argument('--wandb_project', type=str, default='segmentation_practice')
    parser.add_argument('--wandb_entity', type=str, default='cv_1')
    parser.add_argument('--wandb_run', type=str, default='exp')

    args = parser.parse_args()
    
    # 모델 sweep 용 모델명으로 이름 지정 -> 필요없으면 지워도됨
    args.wandb_run += "_" + args.segmentation_model + '_' + args.encoder_name + '_' + args.encoder_weights

    # early stop 안쓰는 경우 patience를 num_epochs으로 설정
    if not args.early_stop:
        args.patience = args.num_epochs

    return args

def seed_everything(seed=2022):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)

def Mixup(images, masks, alpha=0.2):
    # masks: one-hot encoding된 y
    x1, y1 = images[0], masks[0]
    x2, y2 = images[1], masks[1]

    lambda_param = np.random.beta(alpha, alpha)
    images = lambda_param*x1 + (1-lambda_param)*x2
    masks = lambda_param*y1 + (1-lambda_param)*y2

    return images, masks


def train(args):

    seed_everything(args.seed)

    # --settings
    device = args.device

    start_time = datetime.now(timezone('Asia/Seoul')).strftime('_%y%m%d_%H%M%S')
    saved_dir = os.path.join(args.saved_dir, args.wandb_run + start_time)

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    # config 설정
    with open(os.path.join(saved_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent = 4)

    # wandb 설정
    wandb.init(
            project=f'{args.wandb_project}',
            entity=f'{args.wandb_entity}',
            name = args.wandb_run + start_time,
    )
    wandb.config.update({
        "run_name": args.wandb_run,
        "device": args.device,
        "learning_rate": args.learning_rate, 
        "train_batch_size":args.train_batch_size,
        "valid_batch_size":args.valid_batch_size,
        "criterion":args.criterion,
        "optimizer":args.optimizer,
        "max_epoch":args.num_epochs,
        "val_every": args.val_every,
        "seed": args.seed,
        "wandb_project":args.wandb_project,
        "wandb_entity":args.wandb_entity
    })

    train_transform = A.Compose([
                                A.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.5),
                                A.Normalize(mean=[0.46009142, 0.43957697, 0.41827273], std=[0.21060736, 0.20755924, 0.21633709],
                                           max_pixel_value=1.0),
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                            A.Normalize(mean=[0.46009142, 0.43957697, 0.41827273], std=[0.21060736, 0.20755924, 0.21633709],
                                       max_pixel_value=1.0),
                            ToTensorV2()
                            ])


    # --dataset
    train_dataset = CustomDataLoader(data_dir=args.train_path, mode='train', transform=train_transform)

    # validation dataset
    valid_dataset = CustomDataLoader(data_dir=args.valid_path, mode='val', transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.train_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                            batch_size=args.valid_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)

    # --model
    model_module = getattr(import_module("segmentation_models_pytorch"), args.segmentation_model)
    model = model_module(
        encoder_name=args.encoder_name,       # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=args.encoder_weights, # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,                           # model output channels (number of classes in your dataset)
    )

    # --criterion
    if args.criterion == 'CrossEntropyLoss':
        criterion = getattr(import_module("torch.nn"), args.criterion)()
    else:
        criterion = getattr(import_module("segmentation_models_pytorch.losses"), args.criterion)()

    # --optimizer
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    # scheduler = MultiStepLR(optimizer, milestones=[args.num_epochs // 2], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, last_epoch=-1)
                                                           
    scaler = torch.cuda.amp.GradScaler()

    # Early Stopping 변수
    counter = 0

    #print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    best_mIoU = -999999

    categories = [
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
    
    num_train_batches = math.ceil(len(train_dataset) / args.train_batch_size)

    with open(os.path.join(saved_dir,'log.txt'), 'w') as f:
        for epoch in range(args.num_epochs):

            model.train()

            hist = np.zeros((n_class, n_class))
            train_loss = 0

            with tqdm(total=num_train_batches) as pbar:
                for step, (images, masks, _) in enumerate(train_loader):
                    pbar.set_description(f'[Train] Epoch [{epoch+1}/{args.num_epochs}]')

                    optimizer.zero_grad()

                    if args.mixup:
                        images, masks = Mixup(images, masks, alpha=args.alpha)

                    images = torch.stack(images)       
                    masks = torch.stack(masks).long() 
                    
                    images, masks = images.to(device), masks.to(device)
                    model = model.to(device)

                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
                    
                    outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                    masks = masks.detach().cpu().numpy()
                    
                    hist = add_hist(hist, masks, outputs, n_class=n_class)
                    acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                    
                    pbar.update(1)
                    
                    pbar.set_postfix({
                        'Loss' : round(loss.item(),4), 
                        'Accuracy' : round(acc, 4),
                        'mIoU' : round(mIoU,4)
                        })
                    
                    train_loss += loss.item()
            
            scheduler.step()
            
            train_log = '[EPOCH TRAIN {}/{}] : Train Loss {} - Train Accuracy {} - Train mIoU {}'.format(epoch+1, 
                                                                                                        args.num_epochs, 
                                                                                                        round(train_loss / len(train_loader), 4),
                                                                                                        round(acc, 4),
                                                                                                        round(mIoU, 4))
            print(train_log)
            f.write(train_log + '\n')

            wandb.log({
                "train/loss": round(train_loss / len(train_loader),4),
                "train/mIoU": round(mIoU,4),
                "train/accuracy" : round(acc, 4),
                "train/learning_rate" : get_lr(optimizer)
            })

            save_model(model, saved_dir, 'latest.pt')
                                        
            # validation 주기에 따른 loss 출력 및 best model 저장
            if (epoch + 1) % args.val_every == 0:

                #print(f'Start validation #{epoch + 1}')
                model.eval()

                with torch.no_grad():
                    n_class = 11
                    total_loss = 0
                    cnt = 0
                    
                    hist = np.zeros((n_class, n_class))

                    num_valid_batches = math.ceil(len(valid_dataset) / args.valid_batch_size)
                    with tqdm(total=num_valid_batches) as pbar:
                        for step, (images, masks, _) in enumerate(valid_loader):
                            pbar.set_description(f'[Valid] Epoch [{epoch+1}/{args.num_epochs}]')
                            
                            images = torch.stack(images)       
                            masks = torch.stack(masks).long()  

                            images, masks = images.to(device), masks.to(device)            
                            
                            # device 할당
                            model = model.to(device)
                            
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                            total_loss += loss
                            cnt += 1
                            
                            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                            masks = masks.detach().cpu().numpy()
                            
                            hist = add_hist(hist, masks, outputs, n_class=n_class)
                            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

                            pbar.update(1)

                            pbar.set_postfix({
                                'Loss': round(total_loss.item() / cnt, 4), 
                                'Accuracy' : round(acc, 4),
                                'mIoU' : round(mIoU,4)
                                })
                    
                    #IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , sorted_df['Categories'])]
                    IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , categories)]
                    IoU_by_class_logging = [[classes, round(IoU,4)] for IoU, classes in zip(IoU , categories)]
                    IoU_by_class_table = wandb.Table(data=IoU_by_class_logging, columns=['label','value'])
                    
                    avrg_loss = total_loss.item() / cnt
                    valid_log = '[EPOCH VALID {}/{}] : Valid Loss {} - Valid Accuracy {} - Valid mIoU {}'.format(epoch+1, 
                                                                                                                args.num_epochs, 
                                                                                                                round(avrg_loss, 4),
                                                                                                                round(acc, 4),
                                                                                                                round(mIoU, 4))
                    print(valid_log)
                    f.write(valid_log + '\n')
                        
                    wandb.log({
                        "valid/Average Loss": round(avrg_loss, 4),
                        "valid/Accuracy": round(acc, 4),
                        "valid/mIoU": round(mIoU, 4),
                        "valid/IoU_by_class_table" : wandb.plot.bar(IoU_by_class_table, 'label','value', title='IoU By Class') 
                    })

                if avrg_loss < best_loss:
                    print(f"!!! Best Loss at epoch: {epoch + 1} !!!")
                    f.write(f"!!! Best Loss at epoch: {epoch + 1} !!!\n")
                    best_loss = avrg_loss
                    save_model(model, saved_dir, 'best_loss.pt')

                if best_mIoU < mIoU:
                    print(f"!!! Best mIoU at epoch: {epoch + 1} !!!")
                    f.write(f"!!! Best mIoU at epoch: {epoch + 1} !!!\n")
                    best_mIoU = mIoU
                    save_model(model, saved_dir, 'best_mIoU.pt')
                    counter = 0
                else:
                    counter += 1

                # patience 횟수 동안 성능 향상이 없으면 학습 종료
                if counter > args.patience:
                    print(f"Early Stopping at epoch {epoch + 1}...")
                    f.write(f"Early Stopping at epoch {epoch + 1}...")
                    f.close()
                    break

def main():
    args = parse_args()
    train(args)

if __name__=='__main__':
    main()