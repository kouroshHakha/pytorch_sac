

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from torch.utils.data.dataloader import DataLoader

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from osil.data import OsilPM
from osil.nets import TraphormerLightningModule
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
    parser.add_argument('--ctx_size', default=256, type=int)
    parser.add_argument('--trg_size', default=64, type=int)
    parser.add_argument('--mask_rate', default=0.75, type=float)
    # parser.add_argument('--use_wandb', '-wb', action='store_true') # not setup right now
    # parser.add_argument('--ckpt', type=str)
    # parser.add_argument('--resume', action='store_true')
    # parser.add_argument('--num_eval_episodes', default=10, type=int)
    # parser.add_argument('--num_trajs', '-ntraj', default=100, type=int)

    return parser.parse_args()


def main(pargs):
    exp_name = f'osil_pm'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)

    dataset = OsilPM(ctx_size=pargs.ctx_size, trg_size=pargs.trg_size)
    tloader = DataLoader(dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=0)
    c_s, c_a, t_s, t_a = dataset[0]

    # # plot some example datasets
    # _, axes = plt.subplots(3, 4, figsize=(15, 8))
    # axes = axes.flatten()
    #
    # batched_c_s, batched_c_a, batched_t_s, batched_t_a = next(iter(tloader))
    # masks = (np.random.rand(*batched_c_s.shape[:2]) > pargs.mask_rate)
    # masks[:, 0] = True
    # masks[:, -1] = True
    #
    # for i in range(12):
    #     mask = masks[i]
    #     axes[i].plot(batched_t_s[i, :, 0], batched_t_s[i, :, 1], color='b', linestyle='-', linewidth=3.5)
    #     axes[i].plot(batched_c_s[i, :, 0], batched_c_s[i, :, 1], color='b', linestyle='-')
    #     axes[i].scatter(batched_c_s[i, mask, 0], batched_c_s[i, mask, 1], marker='.', color='k', s=25)
    #     # axes[i].scatter([goals[i, 0].item()], [goals[i, 1].item()], marker='*', color='green')
    #     axes[i].scatter(batched_c_s[i, 0, 0], batched_c_s[i, 0, 1], marker='*', color='r', s=100)
    #     axes[i].scatter(batched_c_s[i, -1, 0], batched_c_s[i, -1, 1], marker='*', color='g', s=100)
    #
    # plt.tight_layout()
    # plt.show()

    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        max_ep_len=1024,
        obs_shape=t_s.shape[1:],
        ac_dim=t_a.shape[-1],
        loss_type='only_action',
        mask_rate=pargs.mask_rate,
        lr=1e-4,
    )

    agent = TraphormerLightningModule(config)

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
