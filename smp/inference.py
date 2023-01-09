import os
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import albumentations as A
from tqdm import tqdm

from importlib import import_module
from argparse import ArgumentParser
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import ttach as tta

from dataset import *


def parse_args():
    parser = ArgumentParser()

    # model
    parser.add_argument("--model_path", type=str, default="FPN_mit_b4_aug_230101_192133")
    parser.add_argument("--metric", type=str, default="epoch_67")

    # --tta
    parser.add_argument("--tta", type=bool, default=False)
    parser.add_argument("--merge_mode", type=str, default='mean', help='mean, gmean, sum, max, min, tsharpen')

    # dataset path
    parser.add_argument("--test_path", type=str, default="/opt/ml/input/data/test.json")

    # hyperparameters
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    return args


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def test(args):

    device = args.device

    print("Load pretrained model and Set dataloader")

    model_dir = os.path.join("/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-01/smp/trained_models", args.model_path)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    model_name = cfg["segmentation_model"]
    encoder_name = cfg["encoder_name"]
    encoder_weights = cfg['encoder_weights']

    model_module = getattr(import_module("segmentation_models_pytorch"), model_name)
    model = model_module(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=11,
        #encoder_output_stride=32,
    )
    preprocessing_fn = get_preprocessing_fn(encoder_name, encoder_weights)
    test_transform = get_transform(mode='test', preprocessing_fn=preprocessing_fn)

    ckpt = torch.load(os.path.join(model_dir, args.metric + ".pt"), map_location=device)
    model.load_state_dict(ckpt["net"])
    model = model.to(device)

    test_dataset = CustomDataLoader(data_dir=args.test_path, mode="test", transform=test_transform)
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

    if args.tta:
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            #tta.Rotate90(angles=[0,180]),
            tta.Scale(scales=[1,2,4])
        ])

        tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode=args.merge_mode)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            if args.tta:
                outs = tta_model(torch.stack(imgs).to(device))
            else:
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

    submission.to_csv(os.path.join("/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-01/smp/trained_models", args.model_path, f"{args.metric}_{args.merge_mode if args.tta else 'default'}.csv"), index=False)


def main():
    args = parse_args()
    test(args)


if __name__ == "__main__":
    main()
