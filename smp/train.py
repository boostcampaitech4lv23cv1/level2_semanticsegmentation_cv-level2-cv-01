import os
import math
import json
import warnings
warnings.filterwarnings("ignore")

import torch
from utils import label_accuracy_score, add_hist
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR

import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from importlib import import_module

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb
from pytz import timezone
from datetime import datetime

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from dataset import get_transform, collate_fn, CustomDataLoader, Mixup
from utils import seed_everything, get_lr, save_model, CosineAnnealingWarmUpRestarts

# when using xception encoder
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = ArgumentParser()

    # model
    parser.add_argument(
        "--segmentation_model", type=str, default="FPN", 
        help='Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN'
    )
    parser.add_argument("--encoder_name", type=str, default="mit_b5")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")

    # path
    parser.add_argument("--saved_dir", type=str, default="trained_models")

    # dataset path
    parser.add_argument("--train_path", type=str, default="/opt/ml/input/data/fold5/train_all_sorted_train0.json")
    parser.add_argument("--valid_path", type=str, default="/opt/ml/input/data/fold5/train_all_sorted_val0.json")

    # hyperparameters
    parser.add_argument("--num_epochs", type=int, default=120)  # 20
    parser.add_argument(
        "--criterion", type=str, default="CrossEntropyLoss",
        help='CrossEntropyLoss, JaccardLoss, DiceLoss, FocalLoss, LovaszLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss, TverskyLoss, MCCLoss'
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)  # 1e-4
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--num_workers", type=int, default=4)

    # mixup
    parser.add_argument("--mixup", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.2)

    # early stopping
    parser.add_argument("--early_stop", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=20)

    # settings
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # wandb
    parser.add_argument("--wandb_project", type=str, default="Segmentation")
    parser.add_argument("--wandb_entity", type=str, default="wooyeolbaek")
    parser.add_argument("--wandb_run", type=str, default="exp")

    args = parser.parse_args()

    # 모델 sweep 용 모델명으로 이름 지정 -> 필요없으면 지워도됨
    #args.wandb_run = args.segmentation_model + "_" + args.encoder_name + "_" + args.encoder_weights
    args.wandb_run = args.segmentation_model + "_" + args.encoder_name + "_aug_" + args.train_path.split('/')[-1].split('.')[0] 
    

    # early stop 안쓰는 경우 patience를 num_epochs으로 설정
    if not args.early_stop:
        args.patience = args.num_epochs

    return args

def train(args):

    seed_everything(args.seed)

    # --settings
    device = args.device

    start_time = datetime.now(timezone("Asia/Seoul")).strftime("_%y%m%d_%H%M%S")
    saved_dir = os.path.join(args.saved_dir, args.wandb_run + start_time)

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # config 설정
    with open(os.path.join(saved_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # wandb 설정
    wandb.init(
        project=f"{args.wandb_project}",
        entity=f"{args.wandb_entity}",
        name=args.wandb_run + start_time,
    )
    wandb.config.update(
        {
            "run_name": args.wandb_run,
            "device": args.device,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "valid_batch_size": args.valid_batch_size,
            "criterion": args.criterion,
            "optimizer": args.optimizer,
            "max_epoch": args.num_epochs,
            "val_every": args.val_every,
            "seed": args.seed,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
        }
    )

    # --model
    model_module = getattr(
        import_module("segmentation_models_pytorch"), args.segmentation_model
    )
    model = model_module(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=11,
        #encoder_output_stride=32,
    )
    preprocessing_fn = get_preprocessing_fn(args.encoder_name, args.encoder_weights)
    
    train_transform = get_transform(mode='train', preprocessing_fn=preprocessing_fn)
    val_transform = get_transform(mode='valid', preprocessing_fn=preprocessing_fn)

    # --dataset
    train_dataset = CustomDataLoader(
        data_dir=args.train_path, mode="train", transform=train_transform
    )

    # validation dataset
    valid_dataset = CustomDataLoader(
        data_dir=args.valid_path, mode="val", transform=val_transform
    )

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )


    # --criterion
    if args.criterion == "CrossEntropyLoss":
        criterion = getattr(import_module("torch.nn"), args.criterion)()
    else:
        criterion = getattr(
            import_module("segmentation_models_pytorch.losses"), args.criterion
        )(mode="multiclass")

    # --optimizer
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    #scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.)
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=6, T_mult=3, eta_max=3e-4,  T_up=3, gamma=0.6)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=5, T_mult=3, eta_max=3e-4,  T_up=3, gamma=0.4)

    scaler = torch.cuda.amp.GradScaler()

    # Early Stopping 변수
    counter = 0

    n_class = 11
    best_loss = 9999999
    best_mIoU = -999999

    categories = [
        "Background",
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    ]

    num_train_batches = math.ceil(len(train_dataset) / args.train_batch_size)

    with open(os.path.join(saved_dir, "log.txt"), "w") as f:
        for epoch in range(args.num_epochs):

            model.train()

            hist = np.zeros((n_class, n_class))
            train_loss = 0

            with tqdm(total=num_train_batches) as pbar:
                for step, (images, masks, _) in enumerate(train_loader):
                    pbar.set_description(f"[Train] Epoch [{epoch+1}/{args.num_epochs}]")

                    optimizer.zero_grad()

                    if args.mixup:
                        images, masks = Mixup(images, masks, alpha=args.alpha)

                    images = torch.stack(images).to(device)
                    masks = torch.stack(masks).long().to(device)

                    model = model.to(device)

                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    #optimizer.zero_grad()
                    #loss.backward()
                    #optimizer.step()

                    outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                    masks = masks.detach().cpu().numpy()

                    hist = add_hist(hist, masks, outputs, n_class=n_class)
                    acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

                    pbar.update(1)

                    pbar.set_postfix(
                        {
                            "Loss": round(loss.item(), 4),
                            "Accuracy": round(acc, 4),
                            "mIoU": round(mIoU, 4),
                        }
                    )

                    train_loss += loss.item()

            scheduler.step()

            train_log = "[EPOCH TRAIN {}/{}] : Train Loss {} - Train Accuracy {} - Train mIoU {}".format(
                epoch + 1,
                args.num_epochs,
                round(train_loss / len(train_loader), 4),
                round(acc, 4),
                round(mIoU, 4),
            )
            print(train_log)
            f.write(train_log + "\n")

            wandb.log(
                {
                    "train/loss": round(train_loss / len(train_loader), 4),
                    "train/mIoU": round(mIoU, 4),
                    "train/accuracy": round(acc, 4),
                    "train/learning_rate": get_lr(optimizer),
                }
            )

            # validation 주기에 따른 loss 출력 및 best model 저장
            if (epoch + 1) % args.val_every == 0:

                model.eval()

                with torch.no_grad():
                    total_loss = 0
                    cnt = 0

                    hist = np.zeros((n_class, n_class))

                    num_valid_batches = math.ceil(
                        len(valid_dataset) / args.valid_batch_size
                    )
                    with tqdm(total=num_valid_batches) as pbar:
                        for step, (images, masks, _) in enumerate(valid_loader):
                            pbar.set_description(
                                f"[Valid] Epoch [{epoch+1}/{args.num_epochs}]"
                            )

                            images = torch.stack(images).to(device)
                            masks = torch.stack(masks).long().to(device)

                            # device 할당
                            model = model.to(device)

                            outputs = model(images)
                            loss = criterion(outputs, masks)
                            total_loss += loss
                            cnt += 1

                            outputs = (
                                torch.argmax(outputs, dim=1).detach().cpu().numpy()
                            )
                            masks = masks.detach().cpu().numpy()

                            hist = add_hist(hist, masks, outputs, n_class=n_class)
                            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(
                                hist
                            )

                            pbar.update(1)

                            pbar.set_postfix(
                                {
                                    "Loss": round(total_loss.item() / cnt, 4),
                                    "Accuracy": round(acc, 4),
                                    "mIoU": round(mIoU, 4),
                                }
                            )

                    IoU_by_class = [
                        {classes: round(IoU, 4)}
                        for IoU, classes in zip(IoU, categories)
                    ]
                    IoU_by_class_logging = [
                        [classes, round(IoU, 4)]
                        for IoU, classes in zip(IoU, categories)
                    ]
                    IoU_by_class_table = wandb.Table(
                        data=IoU_by_class_logging, columns=["label", "value"]
                    )

                    avrg_loss = total_loss.item() / cnt
                    valid_log = '[EPOCH VALID {}/{}] : Valid Loss {} - Valid Accuracy {} - Valid mIoU {}\nIoU by class{}'.format(
                        epoch + 1,
                        args.num_epochs,
                        round(avrg_loss, 4),
                        round(acc, 4),
                        round(mIoU, 4),
                        IoU_by_class
                    )
                    print(valid_log)
                    f.write(valid_log + "\n")

                    wandb.log(
                        {
                            "valid/Average Loss": round(avrg_loss, 4),
                            "valid/Accuracy": round(acc, 4),
                            "valid/mIoU": round(mIoU, 4),
                            "valid/IoU_by_class_table": wandb.plot.bar(
                                IoU_by_class_table,
                                "label",
                                "value",
                                title="IoU By Class",
                            ),
                        }
                    )
                
                save_model(model, saved_dir, "latest.pt")
                
                # 50 이상인 5의 배수마다 ckpt 저장
                if epoch + 1 > 65 and (epoch + 1)%10 == 0:
                    save_model(model, saved_dir, f"epoch_{epoch+1}.pt")

                if avrg_loss < best_loss:
                    print(f"!!! Best Loss at epoch: {epoch + 1} !!!")
                    f.write(f"!!! Best Loss at epoch: {epoch + 1} !!!\n")
                    best_loss = avrg_loss
                    save_model(model, saved_dir, "best_loss.pt")

                if best_mIoU < mIoU:
                    print(f"!!! Best mIoU at epoch: {epoch + 1} !!!")
                    f.write(f"!!! Best mIoU at epoch: {epoch + 1} !!!\n")
                    best_mIoU = mIoU
                    save_model(model, saved_dir, "best_mIoU.pt")
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


if __name__ == "__main__":
    main()
