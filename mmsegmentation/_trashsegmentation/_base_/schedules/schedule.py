lr = 1e-4

# optimizer
optimizer = dict(
    type='AdamW', 
    lr=lr, 
    weight_decay=0.01
)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5
)

# runtime settings
runner = dict(
    type='EpochBasedRunner', 
    max_epochs=10
)
checkpoint_config = dict(
    by_epoch=True, 
    interval=1000
)
evaluation = dict(
    interval=1, 
    metric='mIoU', 
    save_best='mIoU', 
    classwise=True
)