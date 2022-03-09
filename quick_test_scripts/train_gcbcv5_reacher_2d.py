
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.nn as nn

from utils import stack_frames, write_pickle, write_yaml, gif_to_tensor_image

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from osil.nets import GCBCv2, GCBCv3, GCBCv4, GCBCv5, get_goal_color
from osil.utils import ParamDict
from osil.eval import EvaluationCallback
from osil.evaluators.reacher2d import EvaluatorReacher2D_GCBC_Img, EvaluatorReacher2D_GCBC_State

from osil.datasets.reacher2d.gcbc import Reacher2DGCBCDataset
from pytorch_lightning.callbacks import ModelSummary

from osil.debug import register_pdb_hook
register_pdb_hook()

import torch
import d4rl; import gym
import envs

import tqdm

import torchvision
from torchvision.utils import draw_bounding_boxes
import einops


def _parse_args():

    parser = argparse.ArgumentParser()
    # basic common params
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight_decay', '-wc', default=1e-4, type=float)
    parser.add_argument('--hidden_dim', '-hd', default=256, type=int)
    parser.add_argument('--goal_enc_dim', '-gd', default=32, type=int)
    parser.add_argument('--batch_size', '-bs', default=1024, type=int)
    parser.add_argument('--lr', '-lr', default=3e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--env_name', type=str)
    # other params
    parser.add_argument('--num_shots', default=-1, type=int, 
                        help='number of shots per each task variation \
                            (-1 means max number of shots available in the dataset)')
    parser.add_argument('--task_size', default=-1, type=int)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--stack_frames', default=4, type=int)
    parser.add_argument('--version', default=3, type=int)
    parser.add_argument('--use_huber_loss', action='store_true')
    parser.add_argument('--use_contrastive_loss', type=float, default=0.0)
    parser.add_argument('--use_hinge_margin_loss', type=float, default=0.0)
    parser.add_argument('--use_target_eef_loss', type=float, default=0.0)
    parser.add_argument('--use_target_color_loss', type=float, default=0.0)
    parser.add_argument('--enc_type', type=str, default='resnet')

    # parser.add_argument('--gd', '-gd', default=-1, type=int)
    # checkpoint resuming and testing
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--eval_every_nsteps', default=1000, type=int)
    parser.add_argument('--start_eval_after', default=-1, type=int)

    # wandb
    parser.add_argument('--use_wandb', '-wb', action='store_true')
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')
    parser.add_argument('--num_workers', default=0, type=int)

    return parser.parse_args()

def main(pargs):
    exp_name = f'gcbcv5_reacher2d'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    num_workers = pargs.num_workers

    train_path = f'reacher_2d_train_v{pargs.version}'
    test_path = f'reacher_2d_test_v{pargs.version}'

    train_dataset = Reacher2DGCBCDataset(data_path=train_path, nshots_per_task=pargs.num_shots, 
                                        task_size=pargs.task_size, image_based=pargs.image, 
                                        n_stack_frames=pargs.stack_frames)
    test_dataset = Reacher2DGCBCDataset(data_path=test_path, image_based=pargs.image,  
                                        n_stack_frames=pargs.stack_frames)

    print('training set size:', len(train_dataset))
    print('test set size:',     len(test_dataset))

    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=num_workers, pin_memory=True)
    vloader = DataLoader(test_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=num_workers, pin_memory=True)
    batch = next(iter(tloader))
    obs, goal, act = batch['obs'][0], batch['goal'][0], batch['action'][0]
    # obs_imgs, goal_imgs, acts, aux_gts, goal_augs = next(iter(tloader))
    # obs, goal, act, aux_gt = obs_imgs[0], goal_imgs[0], acts[0], aux_gts[0]
    # s = time.time()
    # for batch in tqdm.tqdm(tloader):
    #     batch = [item.to(pargs.device) for item in batch]
    #     pass
    # print(f'time of data loading with {pargs.num_workers} workers: {time.time() - s}')
    # breakpoint()

    # #### visualizing matching of obs_imgs and goal_imgs
    # import torchvision
    # import einops
    # # grid_imgs = einops.rearrange([obs_imgs[:4], goal_imgs[:4]], 'i B C H W -> (B i) C H W')
    # grid_imgs = einops.rearrange([goal_augs[:4], goal_imgs[:4]], 'i B C H W -> (B i) C H W')
    # grid_imgs = torchvision.utils.make_grid(grid_imgs, nrow=2, pad_value=255)
    
    # plt.imshow(grid_imgs.permute(1, 2, 0))
    # plt.savefig('test_img_input.png')
    # print(aux_gts[:4])
    # breakpoint()

    # # testing image data against environment
    # from PIL import Image
    # import imageio

    # output_path = Path('./test_gcbc_img_data')
    # output_path.mkdir(parents=True, exist_ok=True)
    
    # imgs = train_dataset.obses
    # conds = train_dataset.conds
    # states = train_dataset.states
    # actions = train_dataset.actions

    # def compare(var_id, cond_id):
    #     cond_idx = np.where(conds[var_id] == cond_id)[0].item()

    #     state = states[var_id][cond_idx]
    #     dataset_imgs = imgs[var_id][cond_idx]
    #     acs = actions[var_id][cond_idx]
    #     playback_imgs = []
    #     env = gym.make('Reacher2D-v1')
    #     env.set_task(var_id, cond_id)
    #     env.reset()
    #     env.set_state(state[0,:2], state[0, 2:4])
    #     for a in acs:
    #         img = env.render('rgb_array')
    #         pil_image = Image.fromarray(img)
    #         pil_image = pil_image.resize((64, 64), Image.ANTIALIAS)
    #         img = np.flipud(np.array(pil_image))
    #         playback_imgs.append(img)
    #         env.step(a)

    #     playback_imgs = np.stack(playback_imgs, 0)

    #     imageio.mimsave(output_path / 'playback.gif', playback_imgs)
    #     imageio.mimsave(output_path / 'dataset.gif', dataset_imgs)

    # # var_id = np.random.randint(len(imgs))
    # var_id = 0
    # cond_id = 9

    # compare(var_id, cond_id)
    # breakpoint()

    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        obs_shape=obs.shape,
        goal_shape=goal.shape,
        ac_dim=act.shape[-1],
        lr=pargs.lr,
        wd=pargs.weight_decay,
        goal_enc_dim=pargs.goal_enc_dim,
        use_huber_loss=pargs.use_huber_loss,
        use_contrastive_loss=pargs.use_contrastive_loss,
        use_target_eef_loss=pargs.use_target_eef_loss,
        use_target_color_loss=pargs.use_target_color_loss,
        enc_type=pargs.enc_type,
    )

    ######## check model's input to output dependency
    args_var = vars(pargs)
    ckpt = args_var.get('ckpt', '')
    resume = args_var.get('resume', False)

    agent = GCBCv5.load_from_checkpoint(ckpt) if ckpt else GCBCv5(config)
    agent = agent.to(device=pargs.device)

    evaluator_cls = EvaluatorReacher2D_GCBC_Img if agent.is_goal_image else EvaluatorReacher2D_GCBC_State


    # # testing the evaluator
    # test_evaluator = evaluator_cls(
    #     ParamDict(max_eval_episodes=1, env_name='Reacher2D-v1', is_image=pargs.image),
    #     train_dataset, mode='train'
    # )

    # test_evaluator.eval(agent)

    # testing the loss computation
    # agent.train()
    # losses = []
    # for batch in tqdm.tqdm(tloader):
    #     batch = {k: v.to(agent.device) for k, v in batch.items()}
    #     ret = agent.training_step([batch])
    #     ret['loss'].backward()
    #     losses.append(ret['loss'].item())
    # print(f'At intialization loss = {np.mean(losses)}')
    # breakpoint()


    train = (ckpt and resume) or not ckpt
    if pargs.use_wandb and train:
        import wandb
        run_name = exp_name if not pargs.run_name else f'{exp_name}_{pargs.run_name}'
        wandb_run = wandb.init(
            project='osil',
            name=run_name,
            dir='./wandb_logs',
            id=pargs.wandb_id,
            resume='allow',
            config=dict(seed=pargs.seed),
        )
        logger = WandbLogger(experiment=wandb_run, save_dir='./wandb_logs')
    else:
        logger = TensorBoardLogger(save_dir='tb_logs', name=exp_name)

    # setting up the call backs
    ckpt_callback_valid = ModelCheckpoint(
                monitor='valid_loss_epoch',
                filename='cgl-{step}-{valid_loss_epoch:.4f}-{epoch:02d}',
                save_last=True,
                save_on_train_epoch_end=True,
                mode='min',
            )

    ckpt_callback_train = ModelCheckpoint(
            monitor='train_loss_epoch',
            filename='cgl-{step}-{train_loss_epoch:.4f}-{epoch:02d}',
            save_last=True,
            save_on_train_epoch_end=True,
            mode='min',
        )

    # evaluation callbacks
    train_evaluator = evaluator_cls(
        ParamDict(max_eval_episodes=10 if pargs.image else 100, 
        env_name='Reacher2D-v1', is_image=pargs.image),
        train_dataset, mode='train'
    )
    eval_ckpt_train = EvaluationCallback(
        train_evaluator, 
        eval_every_n_updates=pargs.eval_every_nsteps, 
        dirpath=ckpt_callback_valid.dirpath,
        start_evaluating_after=pargs.start_eval_after,
    )

    test_evaluator = evaluator_cls(
        ParamDict(max_eval_episodes=10 if pargs.image else 100, 
        env_name='Reacher2DTest-v1', is_image=pargs.image),
        test_dataset, mode='test',
    )
    eval_ckpt_test = EvaluationCallback(test_evaluator, 
        eval_every_n_updates=pargs.eval_every_nsteps,
        dirpath=ckpt_callback_valid.dirpath,
        start_evaluating_after=pargs.start_eval_after,
    )


    trainer = pl.Trainer(
        max_epochs=pargs.max_epochs,
        max_steps=pargs.max_steps,
        resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
        callbacks=[ckpt_callback_valid, ckpt_callback_train, eval_ckpt_train, eval_ckpt_test, ModelSummary(max_depth=-1)],
        # track_grad_norm=2,
    )

    eval_output_dir = ''
    if train:
        trainer.fit(agent, train_dataloaders=[tloader], val_dataloaders=[vloader])
        eval_output_dir = Path(eval_ckpt_test.best_model_path).parent
        print(f'Evaluating with {eval_ckpt_test.best_model_path} ...')
        agent = agent.load_from_checkpoint(eval_ckpt_test.best_model_path)
    else:
        eval_output_dir = pargs.eval_path
        if ckpt and not pargs.eval_path:
            eval_output_dir = str(Path(ckpt).parent.resolve())
            warnings.warn(f'Checkpoint is given for evaluation, but evaluation path is not determined. Using {eval_output_dir} by default')
    

    conf = ParamDict(max_eval_episodes=100, max_render=10, verbose=True, env_name='Reacher2D-v1', is_image=pargs.image)
    evaluator_cls(conf, train_dataset, eval_output_dir, mode='train').eval(agent)
    conf = ParamDict(max_eval_episodes=100, max_render=10, verbose=True, env_name='Reacher2DTest-v1', is_image=pargs.image)
    evaluator_cls(conf, test_dataset, eval_output_dir, mode='test').eval(agent)

    # # testing the eef xy prediction
    # pl.seed_everything(pargs.seed)
    # obs_imgs, goal_imgs, _, target_eef = next(iter(vloader))

    # agent.eval()
    # goal_imgs = goal_imgs
    # _, pred_eef = agent(obs_imgs.to(agent.device), goal_imgs.to(agent.device))
    # eefs = pred_eef.detach().cpu()
    # eefs[:, 1] = -eefs[:, 1]
    # centers = ((eefs / 0.6 + 0.5) * 64).int()
    # dw = 1
    # boxes = torch.stack([centers[:, 0] - dw, centers[:, 1] - dw,  centers[:, 0] + dw, centers[:, 1] + dw], -1)
    # boxes = boxes.clip_(0, 63)
    # obs_imgs_marked = torch.stack([draw_bounding_boxes(img, box[None], fill=True, colors='white') for img, box in zip(obs_imgs, boxes)], 0)

    # for i in range(10):

    #     grid_imgs = einops.rearrange([obs_imgs_marked[i*4:4+i*4], goal_imgs[i*4:4 + 4*i]], 'i B C H W -> (B i) C H W')
    #     grid_imgs = torchvision.utils.make_grid(grid_imgs, nrow=2, pad_value=255)

    #     plt.imshow(grid_imgs.permute(1, 2, 0))
    #     plt.savefig(f'img_eef_test_{i}.png')

if __name__ == '__main__':
    main(_parse_args())
