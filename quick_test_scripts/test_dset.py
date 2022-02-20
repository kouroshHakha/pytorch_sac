from osil.data import OsilPairedDataset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data_path = Path('./maze2d-open-v0_osil_short_trajs_v2')
tset = OsilPairedDataset(data_path, mode='train')
vset = OsilPairedDataset(data_path, mode='valid')
test_set = OsilPairedDataset(data_path, mode='test')

def get_last_xys(dset):
    last_xys = []
    for task_id, v_id in dset.allowed_ids:
        episodes = dset.raw_data[task_id][v_id]
        last_xys.append(np.stack([ep['state'][-1][:2] for ep in episodes], 0))
    last_xys = np.concatenate(last_xys, 0)
    return last_xys

def get_first_xys(dset):
    fist_xys = []
    for task_id, v_id in dset.allowed_ids:
        episodes = dset.raw_data[task_id][v_id]
        fist_xys.append(np.stack([ep['state'][0][:2] for ep in episodes], 0))
    fist_xys = np.concatenate(fist_xys, 0)
    return fist_xys

train_last_xys = get_last_xys(tset)
valid_last_xys = get_last_xys(vset)
test_last_xys = get_last_xys(test_set)

plt.close()
plt.scatter(train_last_xys[:, 0], train_last_xys[:, 1], label='train', s=3, alpha=0.5)
plt.scatter(valid_last_xys[:, 0], valid_last_xys[:, 1], label='valid', s=3, alpha=0.5)
plt.scatter(test_last_xys[:, 0], test_last_xys[:, 1], label='test', s=3, alpha=0.5)
plt.xlim([0, 4])
plt.ylim([0, 6])
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))
plt.savefig(data_path / 'train_valid_goals.png')

train_first_xys = get_first_xys(tset)
valid_first_xys = get_first_xys(vset)
test_first_xys = get_first_xys(test_set)

plt.close()
plt.scatter(train_first_xys[:, 0], train_first_xys[:, 1], label='train', s=3, alpha=0.5)
plt.scatter(valid_first_xys[:, 0], valid_first_xys[:, 1], label='valid', s=3, alpha=0.5)
plt.scatter(test_first_xys[:, 0], test_first_xys[:, 1], label='test', s=3, alpha=0.5)
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))
plt.xlim([0, 4])
plt.ylim([0, 6])
plt.savefig(data_path / 'train_valid_rsts.png')


#### plot examples of pairs
print('Plotting examples ...')
T = 16
nrows = int(T ** 0.5)
ncols = -(-T // nrows) # cieling

plt.close()
_, axes = plt.subplots(nrows, ncols, figsize=(15, 8), squeeze=False)
axes = axes.flatten()
for idx in range(T):
    sample = tset[idx]
    demo_xy = sample['context_s'][:, :2]
    target_xy = sample['target_s'][:, :2]

    axes[idx].plot(demo_xy[:, 0], demo_xy[:, 1], label='demo', linewidth=5, color='red', alpha=0.5)
    axes[idx].plot(target_xy[:, 0], target_xy[:, 1], label='target', linewidth=5, color='orange', alpha=0.5)
    axes[idx].scatter([demo_xy[-1, 0]], [demo_xy[-1, 1]], s=320, marker='*', c='green', label='goal')
    axes[idx].scatter([target_xy[-1, 0]], [target_xy[-1, 1]], s=320, marker='*', c='red', label='target_end')
    axes[idx].set_xlim([0, 4])
    axes[idx].set_ylim([0, 6])

plt.legend(loc='best')
plt.savefig(data_path / f'train_examples.png')
