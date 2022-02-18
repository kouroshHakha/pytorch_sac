# Update logs for osil project

To run the gcbc_v2 for goal conditioned behavioral cloning on the pm data for `ns=5`:

```
python quick_test_scripts/train_gcbcv2.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 -ns 5 -wb --run_name ns_5
```

You can look at `gcbcv2_maze2d_open_ns_5_noisy_target` on wandb for a reference. Other experiments should be done similarly:

To run transformer osil (supervised osil) also used for pretraining the embedding:

```
python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots 100 -wb --run_name tosil_maze2d_open_ns_100
```

This will create a log path under `wandb_osil/osil/<WANDB_ID>` where you can find the checkpoints (all the ckpts I have generated so far are under `/shared`).


To run the semi-supervised osil with pretrained embedding:

```
python quick_test_scripts/train_gcbcv2_w_osil_embedding_v3.py --dataset_path ./maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 -bs 128 --max_steps 10000 --encoder_ckpt wandb_logs/osil/38tubxrp/checkpoints/cgl-step=964-valid_loss=0.1720-epoch=964.ckpt --model osil -wb --force_emb -kn 100 --run_name semi_osil_ns5_self_emb_kn=100
```

The rest of experiment commands are included in `run_exps.sh`.


To run the code in docker:

```bash
cd docker
docker build -t osil:latest .
cd ..
docker run -it -v $PWD:/root/projects/osil --workdir /root/projects/osil --rm  osil:latest
```

# Simple and efficient implementations of SAC and DDPG in PyTorch.

This repository provides implementation of several agents for continuous control tasks from the [DeepMind Control Suite](https://github.com/deepmind/dm_control).

<p align="center">
  <img width="19.5%" src="https://i.imgur.com/NzY7Pyv.gif">
  <img width="19.5%" src="https://imgur.com/O5Va3NY.gif">
  <img width="19.5%" src="https://imgur.com/PCOR9Mm.gif">
  <img width="19.5%" src="https://imgur.com/H0ab6tz.gif">
  <img width="19.5%" src="https://imgur.com/sDGgRos.gif">
  <img width="19.5%" src="https://imgur.com/gj3qo1X.gif">
  <img width="19.5%" src="https://imgur.com/FFzRwFt.gif">
  <img width="19.5%" src="https://imgur.com/W5BKyRL.gif">
  <img width="19.5%" src="https://imgur.com/qwOGfRQ.gif">
  <img width="19.5%" src="https://imgur.com/Uubf00R.gif">
 </p>
 

If you use this code in your research project please cite us as:
```
@misc{pytorch_sac,
  author = {Yarats, Denis and Kostrikov, Ilya},
  title = {Soft Actor-Critic (SAC) implementation in PyTorch},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/denisyarats/pytorch_sac}},
}
```

## Implemented Agents
| Agent | Command | Paper |
|---|---|---|---|
| SAC | `agent=sac` |  [paper](https://arxiv.org/pdf/1801.01290.pdf)|
| DDPG | `agent=ddpg` |  [paper](https://arxiv.org/pdf/1509.02971.pdf)|


## Requirements

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install dependencies
```sh
conda env create -f conda_env.yml
conda activate pytorch_sac
```

## Instructions
To train SAC on Walker Walk run
```
python train.py agent=sac task=walker_walk
```
To train DDPG on Jaco Reach Duplo run 
```
python train.py agent=ddpg task=jaco_reach_duplo
```

## Monitoring
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```
The console output is also available in a form:
```
| train | F: 6000 | S: 6000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```


## License
The majority of this repository is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.
