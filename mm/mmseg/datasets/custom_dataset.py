import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .pipelines import Compose, LoadAnnotations
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyCustomDataset(CustomDataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        gt_seg_map_loader_cfg (dict, optional): build LoadAnnotations to
            load gt for evaluation, load from disk by default. Default: None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    CLASSES = (
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
    )

    PALETTE = [
        [0, 192, 64],
        [0, 192, 64],
        [0, 64, 96],
        [128, 192, 192],
        [0, 64, 64],
        [0, 192, 224],
        [0, 192, 192],
        [128, 192, 64],
        [0, 192, 96],
        [128, 192, 64],
        [128, 32, 192],
        [0, 0, 224],
        [0, 0, 64],
        [0, 160, 192],
        [128, 0, 96],
        [128, 0, 192],
        [0, 32, 192],
        [128, 128, 224],
        [0, 0, 192],
        [128, 160, 192],
        [128, 128, 0],
        [128, 0, 32],
        [128, 32, 0],
        [128, 0, 128],
        [64, 128, 32],
        [0, 160, 0],
        [0, 0, 0],
        [192, 128, 160],
        [0, 32, 0],
        [0, 128, 128],
        [64, 128, 160],
        [128, 160, 0],
        [0, 128, 0],
        [192, 128, 32],
        [128, 96, 128],
        [0, 0, 128],
        [64, 0, 32],
        [0, 224, 128],
        [128, 0, 0],
        [192, 0, 160],
        [0, 96, 128],
        [128, 128, 128],
        [64, 0, 160],
        [128, 224, 128],
        [128, 128, 64],
        [192, 0, 32],
        [128, 96, 0],
        [128, 0, 192],
        [0, 128, 32],
        [64, 224, 0],
        [0, 0, 64],
        [128, 128, 160],
        [64, 96, 0],
        [0, 128, 192],
        [0, 128, 160],
        [192, 224, 0],
        [0, 128, 64],
        [128, 128, 32],
        [192, 32, 128],
        [0, 64, 192],
        [0, 0, 32],
        [64, 160, 128],
        [128, 64, 64],
        [128, 0, 160],
        [64, 32, 128],
        [128, 192, 192],
        [0, 0, 160],
        [192, 160, 128],
        [128, 192, 0],
        [128, 0, 96],
        [192, 32, 0],
        [128, 64, 128],
        [64, 128, 96],
        [64, 160, 0],
        [0, 64, 0],
        [192, 128, 224],
        [64, 32, 0],
        [0, 192, 128],
        [64, 128, 224],
        [192, 160, 0],
        [0, 192, 0],
        [192, 128, 96],
        [192, 96, 128],
        [0, 64, 128],
        [64, 0, 96],
        [64, 224, 128],
        [128, 64, 0],
        [192, 0, 224],
        [64, 96, 128],
        [128, 192, 128],
        [64, 0, 224],
        [192, 224, 128],
        [128, 192, 64],
        [192, 0, 96],
        [192, 96, 0],
        [128, 64, 192],
        [0, 128, 96],
        [0, 224, 0],
        [64, 64, 64],
        [128, 128, 224],
        [0, 96, 0],
        [64, 192, 192],
        [0, 128, 224],
        [128, 224, 0],
        [64, 192, 64],
        [128, 128, 96],
        [128, 32, 128],
        [64, 0, 192],
        [0, 64, 96],
        [0, 160, 128],
        [192, 0, 64],
        [128, 64, 224],
        [0, 32, 128],
        [192, 128, 192],
        [0, 64, 224],
        [128, 160, 128],
        [192, 128, 0],
        [128, 64, 32],
        [128, 32, 64],
        [192, 0, 128],
        [64, 192, 32],
        [0, 160, 64],
        [64, 0, 0],
        [192, 192, 160],
        [0, 32, 64],
        [64, 128, 128],
        [64, 192, 160],
        [128, 160, 64],
        [64, 128, 0],
        [192, 192, 32],
        [128, 96, 192],
        [64, 0, 128],
        [64, 64, 32],
        [0, 224, 192],
        [192, 0, 0],
        [192, 64, 160],
        [0, 96, 192],
        [192, 128, 128],
        [64, 64, 160],
        [128, 224, 192],
        [192, 128, 64],
        [192, 64, 32],
        [128, 96, 64],
        [192, 0, 192],
        [0, 192, 32],
        [64, 224, 64],
        [64, 0, 64],
        [128, 192, 160],
        [64, 96, 64],
        [64, 128, 192],
        [0, 192, 160],
        [192, 224, 64],
        [64, 128, 64],
        [128, 192, 32],
        [192, 32, 192],
        [64, 64, 192],
        [0, 64, 32],
        [64, 160, 192],
        [192, 64, 64],
        [128, 64, 160],
        [64, 32, 192],
        [192, 192, 192],
        [0, 64, 160],
        [192, 160, 192],
        [192, 192, 0],
        [128, 64, 96],
        [192, 32, 64],
        [192, 64, 128],
        [64, 192, 96],
        [64, 160, 64],
        [64, 64, 0],
    ]

    def __init__(
        self,
        pipeline=None,
        img_dir="/opt/ml/input/data/batch_02_vt/img_dir",
        img_suffix=".jpg",
        ann_dir=None,
        seg_map_suffix=".jpg",
        split=None,
        data_root=None,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        gt_seg_map_loader_cfg=None,
        file_client_args=dict(backend="disk"),
    ):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)
        self.gt_seg_map_loader = (
            LoadAnnotations()
            if gt_seg_map_loader_cfg is None
            else LoadAnnotations(**gt_seg_map_loader_cfg)
        )

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert (
                self.CLASSES is not None
            ), "`cls.CLASSES` or `classes` should be specified when testing"

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(
            self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split
        )

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)
