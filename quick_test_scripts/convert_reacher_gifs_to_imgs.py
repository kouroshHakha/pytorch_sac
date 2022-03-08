
from utils import gif_to_tensor_image
from pathlib import Path
from osil.data import OsilDataCollector
import torch

from utils import write_hdf5

import tqdm

datapath = Path('reacher_2d_train_v3')
raw_data = OsilDataCollector.load(datapath).data
output_path = datapath / 'torch_imgs'
output_path.mkdir(exist_ok=True, parents=True)

for task_id in raw_data:
    imgs = []
    for var_id in tqdm.tqdm(sorted(raw_data[task_id].keys())):
        for ep in raw_data[task_id][var_id]:
            gif_file = datapath / 'gifs' / f'color_{var_id}' / f'cond_{ep["cond_id"].item()}.gif'
            traj_imgs= gif_to_tensor_image(gif_file)
            torch.save(traj_imgs, output_path / f'{var_id}_{ep["cond_id"].item()}.torch')

# output_path = datapath / 'numpy_imgs'
# output_path.mkdir(exist_ok=True, parents=True)
# for file in tqdm.tqdm((datapath / 'torch_imgs').iterdir()):

#     if file.suffix == '.torch':
#         traj_imgs = torch.load(file)
#         write_hdf5(dict(data=traj_imgs.numpy()), output_path / (file.stem + '.h5'))
