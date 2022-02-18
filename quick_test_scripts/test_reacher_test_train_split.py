from osil_gen_data.data_collector import OsilDataCollector
import numpy as np
import matplotlib.pyplot as plt

collector = OsilDataCollector.load('reacher_7dof-v1_osil_dataset_v2')
raw_data = collector.data


# for task_id in raw_data:
#     for var_id in raw_data[task_id]:
centers = []
for j in range(64):
    centers.append(np.stack([raw_data[0][j][i]['state'][-1, -3:] for i in range(100)]).mean(0))
centers = np.stack(centers)

splits = {
    'train': np.arange(12, 64).tolist(),
    'valid': [6, 1, 5, 8, 0, 11],
    'test': [4, 10, 3, 9, 7, 2],
}

test = centers[splits['test']]
valid = centers[splits['valid']]
train = centers[splits['train']]

ax = plt.axes(projection ="3d")
ax.scatter3D(test[:, 0],  test[:, 1],  test[:, 2], color='green', alpha=0.8)
ax.scatter3D(valid[:, 0], valid[:, 1], valid[:, 2], color='orange', alpha=0.8)
ax.scatter3D(train[:, 0], train[:, 1], train[:, 2], color='blue', alpha=0.8)
breakpoint()