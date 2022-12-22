# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="MMSegWandbHook",
            by_epoch=False,
            interval=1,
            with_step=False,
            init_kwargs=dict(
                entity="kidsarebornstars",
                project="segmentation",
                name="exp",
            ),
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=10,
        ),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1), ("val", 1)]
cudnn_benchmark = True
