from typing import List

import random
import re
import time

import numpy as np
import torch
import torch.nn as nn


from typing import Any, Union, Dict, Mapping

from pathlib import Path

from ruamel.yaml import YAML
import pickle
import h5py
import numpy as np
from numbers import Number

from PIL import Image
import imageio

import matplotlib.pyplot as plt

PathLike = Union[str, Path]
yaml = YAML(typ='safe')

def gif_to_tensor_image(gif_file):
    gif_obj = imageio.read(gif_file, pilmode='RGB')

    np_frames = []
    for frame in gif_obj:
        pil_image = Image.fromarray(frame)
        pil_image = pil_image.resize((64, 64), Image.ANTIALIAS)
        np_frames.append(np.array(pil_image))
    
    torch_frames = torch.from_numpy(np.stack(np_frames, 0))
    return torch_frames


def save_as_gif(imgs: List[np.ndarray], name: str) -> None:
    imgs_list = [Image.fromarray(img.astype(np.uint8)) for img in imgs]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs_list[0].save(
        name,
        save_all=True,
        append_images=imgs_list[1:],
        duration=100,
        loop=0,
    )


def stack_frames(frames, num_stacks=1):
    padding_shape = (num_stacks-1,) + frames.shape[1:]
    frames = torch.cat([torch.zeros(padding_shape, device=frames.device), frames], 0)
    # N, C, H, W or N, H, W, C
    separated_frames = [list(frames[i-num_stacks+1:i+1]) for i in range(num_stacks-1, len(frames))]
    multi_channel_frames = [torch.cat(stack_of_frames, 0) for stack_of_frames in separated_frames]
    return torch.stack(multi_channel_frames, 0)

def read_yaml(fname: Union[str, Path]) -> Any:
    """Read the given file using YAML.
    Parameters
    ----------
    fname : str
        the file name.
    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    with open(fname, 'r') as f:
        content = yaml.load(f)

    return content

def write_yaml(fname: Union[str, Path], obj: object, mkdir: bool = True) -> None:
    """Writes the given object to a file using YAML format.
    Parameters
    ----------
    fname : Union[str, Path]
        the file name.
    obj : object
        the object to write.
    mkdir : bool
        If True, will create parent directories if they don't exist.
    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    if isinstance(fname, str):
        fpath = Path(fname)
    else:
        fpath = fname

    if mkdir:
        fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(fpath, 'w') as f:
        yaml.dump(obj, f)

def get_full_name(name: str, prefix: str = '', suffix: str = ''):
    """Returns a full name given a base name and prefix and suffix extensions
    Parameters
    ----------
    name: str
        the base name.
    prefix: str
        the prefix (default='')
    suffix
        the suffix (default='')
    Returns
    -------
    full_name: str
        the fullname
    """
    if prefix:
        name = f'{prefix}_{name}'
    if suffix:
        name = f'{name}_{suffix}'
    return name

def read_pickle(fname: Union[str, Path]) -> Any:
    """Read the given file using Pickle.
    Parameters
    ----------
    fname : str
        the file name.
    Returns
    -------
    content : Any
        the object returned by pickle.
    """
    with open(fname, 'rb') as f:
        content = pickle.load(f)

    return content

def write_pickle(fname: Union[str, Path], obj: object, mkdir: bool = True) -> None:
    """Writes the given object to a file using pickle format.
    Parameters
    ----------
    fname : Union[str, Path]
        the file name.
    obj : object
        the object to write.
    mkdir : bool
        If True, will create parent directories if they don't exist.
    Returns
    -------
    content : Any
        the object returned by pickle.
    """
    if isinstance(fname, str):
        fpath = Path(fname)
    else:
        fpath = fname

    if mkdir:
        fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def read_hdf5(fpath: PathLike) -> Dict[str, Any]:
    def _load_hdf5_helper(root: h5py.Group) -> Dict[str, Any]:
        init_dict = {}
        for k, v in root.items():
            if isinstance(v, h5py.Dataset):
                init_dict[k] = np.array(v)
            elif isinstance(v, h5py.Group):
                init_dict[k] = _load_hdf5_helper(v)
            else:
                raise ValueError(f'Does not support type {type(v)}')
        return init_dict

    with h5py.File(fpath, 'r') as f:
        return _load_hdf5_helper(f)



def write_hdf5(data_dict: Mapping[str, Any], fpath: PathLike) -> None:

    def _save_as_hdf5_helper(obj: Mapping[str, Union[Mapping, np.ndarray]], root: h5py.File):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                root.create_dataset(name=k, data=v)
            elif isinstance(v, dict):
                grp = root.create_group(name=k)
                _save_as_hdf5_helper(v, grp)
            elif isinstance(v, Number):
                root.create_dataset(name=k, data=v)
            else:
                raise ValueError(f'Does not support type {type(v)}')

    with h5py.File(fpath, 'w') as root:
        _save_as_hdf5_helper(data_dict, root)
        
class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.lerp_(param.data, tau)


def set_requires_grad(net, value):
    for param in net.parameters():
        param.requires_grad_(value)


def to_torch(xs, device, dtype=None):
    if dtype is not None:
        xs = (x.astype(dtype) for x in xs)
    return tuple(torch.as_tensor(x, device=device, dtype=torch.float) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)
