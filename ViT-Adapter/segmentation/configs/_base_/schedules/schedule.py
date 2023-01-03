# optimizer
optimizer = dict(
    type="AdamW",
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_tab.le": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
optimizer_config = dict()


# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    min_lr=0.0,
    by_epoch=False,
)

# runtime settings
runner = dict(type="IterBasedRunner", max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(
    interval=8000,
    metric="mIoU",
    save_best="mIoU",
    classwise=True,
    pre_eval=True,
)
