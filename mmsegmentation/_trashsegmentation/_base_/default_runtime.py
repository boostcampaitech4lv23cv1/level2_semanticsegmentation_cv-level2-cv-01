from pytz import timezone
from datetime import datetime

exp_name = "Swin+UPerNet"
start_time = datetime.now(timezone("Asia/Seoul")).strftime("_%y%m%d_%H%M%S")

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            interval=50,
            init_kwargs=dict(
                entity="kidsarebornstars",
                project="segmentation",
                name=f"{exp_name}+{start_time}",
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
