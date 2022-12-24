import os
import math
import json
import random
import warnings

warnings.filterwarnings("ignore")

import torch
from utils import label_accuracy_score, add_hist
from importlib import import_module

import numpy as np
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from argparse import ArgumentParser

import segmentation_models_pytorch as smp

from dataset import *


def parse_args():
    parser = ArgumentParser()
    # model
    parser.add_argument(
        "--model_path",
        type=str,
        default="Unet_mit_b5_imagenet_CosineAnnealingWarmRestarts_DiceLoss_221223_234150",
    )
    parser.add_argument(
        "--metric", type=str, default="best_mIoU"
    )  # ['best_mIoU', 'best_loss', 'latest']

    # path
    parser.add_argument("--saved_dir", type=str, default="submission")

    # dataset path
    parser.add_argument("--test_path", type=str, default="/opt/ml/input/data/test.json")

    # hyperparameters
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    return args


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def test(args):

    device = args.device

    print("Load pretrained model and Set dataloader")

    model_dir = os.path.join("trained_models", args.model_path)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    model_name = cfg["segmentation_model"]
    encoder_name = cfg["encoder_name"]

    model_module = getattr(import_module("segmentation_models_pytorch"), model_name)
    model = model_module(
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,  # model output channels (number of classes in your dataset)
    )

    ckpt = torch.load(os.path.join(model_dir, args.metric + ".pt"), map_location=device)
    model.load_state_dict(ckpt["net"])
    model = model.to(device)

    input_size = args.input_size
    test_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.46009142, 0.43957697, 0.41827273],
                std=[0.21060736, 0.20755924, 0.21633709],
                max_pixel_value=1.0,
            ),
            # A.Resize(input_size, input_size),
            ToTensorV2(),
        ]
    )
    test_dataset = CustomDataLoader(
        data_dir=args.test_path, mode="test", transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    size = 256
    transform = A.Compose([A.Resize(size, size)])

    print("Start prediction.")
    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed["mask"]
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i["file_name"] for i in image_infos])

    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    print("Save prediction.")

    # sample_submisson.csv 열기
    submission = pd.read_csv("./submission/sample_submission.csv", index_col=None)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {
                "image_id": file_name,
                "PredictionString": " ".join(str(e) for e in string.tolist()),
            },
            ignore_index=True,
        )

    # submission.csv로 저장
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    submission.to_csv(
        os.path.join(args.saved_dir, args.model_path + "_submission.csv"), index=False
    )


def main():
    args = parse_args()
    test(args)


if __name__ == "__main__":
    main()
