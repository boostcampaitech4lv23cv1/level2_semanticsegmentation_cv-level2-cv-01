lr = 1e-4

# optimizer
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=300,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=5e-6,
)

# runtime settings
runner = dict(type="IterBasedRunner", max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(
    interval=10000,
    metric="mIoU",
    save_best="mIoU",
    classwise=True,
    pre_eval=True,
)
