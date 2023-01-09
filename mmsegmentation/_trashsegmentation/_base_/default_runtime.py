exp_name = "upernet_convnext_xlarge_fp16_640x640_160k_ade20k_albu_230101_JUN"

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="MMSegWandbHook",
            interval=50,
            with_step=False,
            init_kwargs=dict(
                entity="cv_1",
                project="segmentation_practice",
                name=f"{exp_name}",
            ),
        ),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1), ("val", 1)]
opencv_num_threads = 0
mp_start_method = "fork"
cudnn_benchmark = True

seed = 2022
gpu_ids = 0
