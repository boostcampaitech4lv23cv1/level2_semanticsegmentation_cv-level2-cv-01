## Albumentations on mmsegmentation

### How To Use

1. Add this code on **/opt/conda/envs/mm_s/lib/python3.8/site-packages/mmseg/datasets/pipelines/transforms.py**

```
try:
import albumentations
from albumentations import Compose
except ImportError:
albumentations = None
Compose = None

@PIPELINES.register_module()
class Albu(object):
"""Albumentation augmentation. Adds custom transformations from
Albumentations library. Please, visit
`https://albumentations.readthedocs.io` to get more information. An example
of `transforms` is as followed:
.. code-block::
[
dict(
type='ShiftScaleRotate',
shift_limit=0.0625,
scale_limit=0.0,
rotate_limit=0,
interpolation=1,
p=0.5),
dict(
type='RandomBrightnessContrast',
brightness_limit=[0.1, 0.3],
contrast_limit=[0.1, 0.3],
p=0.2),
dict(type='ChannelShuffle', p=0.1),
dict(
type='OneOf',
transforms=[
dict(type='Blur', blur_limit=3, p=1.0),
dict(type='MedianBlur', blur_limit=3, p=1.0)
],
p=0.1),
]
Args:
transforms (list[dict]): A list of albu transformations
keymap (dict): Contains {'input key':'albumentation-style key'}
"""

    def __init__(self, transforms, keymap=None, update_pad_shape=False):
        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.aug = Compose([self.albu_builder(t) for t in self.transforms])
        if not keymap:
            self.keymap_to_albu = {"img": "image", "gt_semantic_seg": "mask"}
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()
        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(f"type must be str, but got {type(obj_type)}")
        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform) for transform in args["transforms"]
            ]
        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.
        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        results = self.aug(**results)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results["pad_shape"] = results["img"].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str
```
2. Add `Albu` on **/opt/conda/envs/mm_s/lib/python3.8/site-packages/mmseg/datasets/pipelines/**init**.py**

```
__ all __ = [.....,'Albu']

```
