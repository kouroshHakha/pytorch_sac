from pathlib import Path
import shutil

dir_list = [
    'wandb_logs/osil/1c8d6br2',
    'wandb_logs/osil/35id8qn0',
    'wandb_logs/osil/1f6w74rb',
    'wandb_logs/osil/2o7hfgrb',
    'wandb_logs/osil/3dw0jrvw',
]

for ckpt_dir in dir_list:
    if Path(ckpt_dir).exists():
        shutil.rmtree(ckpt_dir)
    else:
        print(f'path {ckpt_dir} does not exist.')