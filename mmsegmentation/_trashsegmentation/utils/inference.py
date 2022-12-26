import os
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")

import albumentations as A


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        default="./work_dirs/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k.py",
    )
    parser.add_argument(
        "--test_images_path", type=str, default="/opt/ml/input/data/mmseg/images/test"
    )
    parser.add_argument("--iter", type=str, default="best_mIoU_epoch_52")
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./work_dirs/segformer_mit-b5_512x512_160k_ade20k",
    )
    args = parser.parse_args()
    return args


def test(args):
    cfg = Config.fromfile(args.config_path)
    root = args.test_images_path

    cfg.data.test.img_dir = root
    cfg.data.test.pipeline[1]["img_scale"] = (512, 512)
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 1
    cfg.work_dir = args.work_dir
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    checkpoint_path = os.path.join(cfg.work_dir, f"{args.iter}.pth")

    dataset = build_dataset(cfg.data.test)
    # if len(dataset) != 624:
    #     raise AssertionError(
    #         "Test dataset should be 624 images. Check you test.json file"
    # )
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader)

    submission = pd.read_csv(
        "/opt/ml/input/code/submission/sample_submission.csv", index_col=None
    )
    json_dir = os.path.join("/opt/ml/input/data/test.json")
    with open(json_dir, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)

    input_size = 512
    output_size = 256
    transformed = A.Compose([A.Resize(output_size, output_size)])

    for image_id, predict in enumerate(output):
        image_id = datas["images"][image_id]
        file_name = image_id["file_name"]
        temp_mask = []
        mask = np.array(predict, dtype="uint8")
        mask = transformed(image=mask)
        temp_mask.append(mask["image"])
        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], output_size * output_size]).astype(int)

        string = oms.flatten()
        submission = submission.append(
            {
                "image_id": file_name,
                "PredictionString": " ".join(str(e) for e in string.tolist()),
            },
            ignore_index=True,
        )

    submission.to_csv(
        os.path.join(cfg.work_dir, f"submission_{args.iter}.csv"), index=False
    )

    print("Done")


def main():
    args = parse_args()
    test(args)


if __name__ == "__main__":
    main()
