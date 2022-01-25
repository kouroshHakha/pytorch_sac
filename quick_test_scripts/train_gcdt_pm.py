
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from torch.utils.data.dataloader import DataLoader

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from osil.data import GCPM
from osil.nets import GCDT
from osil.utils import ParamDict

from osil.debug import register_pdb_hook
register_pdb_hook()


def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # parser.add_argument('--weight_decay', '-wc', default=1e-4, type=float)
    parser.add_argument('--hidden_dim', '-hd', default=128, type=int)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    # parser.add_argument('--lr', '-lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--block_size', default=64, type=int)
    # parser.add_argument('--use_wandb', '-wb', action='store_true') # not setup right now
    # parser.add_argument('--ckpt', type=str)
    # parser.add_argument('--resume', action='store_true')
    # parser.add_argument('--num_eval_episodes', default=10, type=int)
    # parser.add_argument('--num_trajs', '-ntraj', default=100, type=int)

    return parser.parse_args()


def main(pargs):
    exp_name = f'gct_pm'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)

    dataset = GCPM(block_size=pargs.block_size)
    tloader = DataLoader(dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=0)
    obs, goal, act = dataset[0]

    # # plot some example datasets
    # _, axes = plt.subplots(3, 4)
    # axes = axes.flatten()
    #
    # states, goals, acts = next(iter(tloader))
    # for i in range(12):
    #     axes[i].plot(states[i, :, 0], states[i, :, 1], color='b', linestyle='--')
    #     axes[i].scatter([goals[i, 0].item()], [goals[i, 1].item()], marker='*', color='green')
    #     axes[i].scatter(states[i, 0, 0], states[i, 0, 1], marker='.', color='r')
    #     axes[i].scatter(states[i, -1, 0], states[i, -1, 1], marker='.', color='g')
    #
    # plt.tight_layout()
    # plt.show()

    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        max_ep_len=1024,
        obs_shape=obs.shape[1:],
        ac_dim=act.shape[-1],
        goal_dim=goal.shape[-1],
        loss_type='only_action',
        lr=1e-4,
    )
    agent = GCDT(config)

    ######## check model's input to output dependency
    # import torch
    # input_tokens = torch.randn(1, 10, pargs.hidden_dim)
    # outputs = agent.transformer(inputs_embeds=input_tokens, output_attentions=True)
    # attn = outputs['attentions'][0][0,0].detach().numpy()
    # plt.imshow(attn)
    # plt.show()

    # ckpt = bc_config.pop('ckpt', '')
    # resume = bc_config.pop('resume', False)
    # bc_config.update(obs_shape=obs.shape[1:], ac_dim=act.shape[-1])
    #
    # agent = BC.load_from_checkpoint(ckpt) if ckpt else BC(bc_config)
    logger = TensorBoardLogger(save_dir='tb_logs', name=exp_name)
    trainer = pl.Trainer(
        max_epochs=pargs.max_epochs,
        # resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
    )

    # eval_output_dir = ''
    # train = (ckpt and resume) or not ckpt
    # if train:
    trainer.fit(agent, train_dataloaders=[tloader])
    # eval_output_dir = Path(trainer.checkpoint_callback.best_model_path).parent


if __name__ == '__main__':
    main(_parse_args())
