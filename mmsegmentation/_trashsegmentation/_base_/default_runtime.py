# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            interval=1000,
            init_kwargs=dict(
                entity="kidsarebornstars",
                project="segmentation",
                name="upernet_beit-large_fp16_8x1_640x640_160k_ade20k",
            ),
        ),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True
