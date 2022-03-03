from pathlib import Path
import shutil

dir_list = [
    'wandb_logs/osil/p1purav8',
]

for ckpt_dir in dir_list:
    if Path(ckpt_dir).exists():
        shutil.rmtree(ckpt_dir)
    else:
        print(f'path {ckpt_dir} does not exist.')