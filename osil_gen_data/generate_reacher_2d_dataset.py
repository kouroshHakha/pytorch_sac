
from importlib.resources import path
from pandas import read_pickle
import envs; import gym

from osil.debug import register_pdb_hook
register_pdb_hook()

import glob
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import imageio
from PIL import Image
import tqdm

from osil_gen_data.data_collector import OsilDataCollector


render = True
mode = 'train'
output_path = f'reacher_2d_{mode}_v3'

if __name__ == '__main__':

    env = gym.make('Reacher2D-v1' if mode=='train' else 'Reacher2DTest-v1')
    # obs = env.reset()
    # obs_dict = env.get_obs_dict()
    # pprint.pprint(obs_dict)
    # breakpoint()


    pkl_files = sorted(glob.glob(str(Path(f'mil_data/reacher_{mode}') / '*.pkl')))
    
    dataset = OsilDataCollector(path=output_path)

    for pkl_file in tqdm.tqdm(pkl_files):
        data = read_pickle(pkl_file)

        task_id = int(Path(pkl_file).stem.split('_')[1])
        gif_dir = Path(output_path) / 'gifs' / f'color_{task_id}'
        gif_dir.mkdir(parents=True, exist_ok=True)
        
        for demo_idx, cond_id in enumerate(data['demoConditions']):
            actions = data['demoU'][demo_idx]
            qpos_init = data['demoX'][demo_idx][0, :2]
            qvel_init = data['demoX'][demo_idx][0, 2:4]

            gif_path = gif_dir / f'cond_{cond_id}.gif'

            env.set_task(task_id, cond_id)
            env.reset()
            obs = env.set_state(qpos_init, qvel_init)

            imgs = []
            obses = [obs]
            for a in actions[:-1]:
                if render:
                    # imgs.append(env.render('rgb_array'))
                    img = env.render('rgb_array')
                    pil_image = Image.fromarray(img)
                    pil_image = pil_image.resize((64, 64), Image.ANTIALIAS)
                    img = np.flipud(np.array(pil_image))
                    imgs.append(img)

                obs, _, _, _ = env.step(a)
                obses.append(obs)

            obses = np.stack(obses, 0)

            if imgs:
                imageio.mimsave(gif_path, imgs)

            ep = dict(state=obses, action=actions, target=env.get_obs_dict()['target_color'], cond_id=np.array(cond_id))
            dataset.append(ep, 0, task_id)
        dataset.save_var(0, task_id)

    dataset.save_meta()
    foo = dataset.npify(dataset.data)
    breakpoint()

    # script fixing old data
    # dataset = OsilDataCollector.load(output_path)
    # pkl_files = sorted(glob.glob(str(Path(f'mil_data/reacher_{mode}') / '*.pkl')))

    # new_dataset = OsilDataCollector(path=f'{output_path}_v2')
    # for pkl_file in tqdm.tqdm(pkl_files):

    #     data = read_pickle(pkl_file)

    #     task_id = int(Path(pkl_file).stem.split('_')[1])
    #     gif_dir = Path(pkl_file).parent / f'color_{task_id}'
    #     gif_dir.mkdir(parents=True, exist_ok=True)
        
    #     for demo_idx, cond_id in enumerate(data['demoConditions']):
    #         episodes = dataset.data[0][task_id]
    #         obses = episodes[demo_idx]['state']
    #         actions = episodes[demo_idx]['action']
    #         target = episodes[demo_idx]['target']
    #         ep = dict(state=obses, action=actions, target=target, cond_id=np.array([cond_id]))
    #         new_dataset.append(ep, 0, task_id)
    #     new_dataset.save_var(0, task_id)

    # new_dataset.save_meta()