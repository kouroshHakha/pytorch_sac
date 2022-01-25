
import argparse
from pathlib import Path
from torch.utils.data.dataloader import DataLoader

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from osil.data import DMC
from osil.nets import BC
from osil.utils import ParamDict
from osil.eval import Evaluator

from osil.debug import register_pdb_hook
register_pdb_hook()


def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--obs_type', default='states', choices=['states', 'pixels'], type=str)
    parser.add_argument('--weight_decay', '-wc', default=1e-4, type=float)
    parser.add_argument('--hidden_dim', '-hd', default=256, type=int)
    parser.add_argument('--batch_size', '-bs', default=256, type=int)
    parser.add_argument('--lr', '-lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--use_wandb', '-wb', action='store_true') # not setup right now
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--num_trajs', '-ntraj', default=100, type=int)

    return parser.parse_args()


def main(pargs):
    exp_name = f'bc_{pargs.task_name}_{pargs.obs_type}_{pargs.num_trajs}'
    print(f'Running {exp_name} ...')
    dataset = DMC(pargs.task_name, pargs.obs_type, block_size=1, num_trajs=pargs.num_trajs)
    tloader = DataLoader(dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=4)
    obs, act = dataset[0]
    
    bc_config = ParamDict(vars(pargs))
    ckpt = bc_config.pop('ckpt', '')
    resume = bc_config.pop('resume', False)
    bc_config.update(obs_shape=obs.shape[1:], ac_dim=act.shape[-1])
    
    agent = BC.load_from_checkpoint(ckpt) if ckpt else BC(bc_config)
    logger = TensorBoardLogger(save_dir='tb_logs', name=exp_name)
    trainer = pl.Trainer(
        max_epochs=pargs.max_epochs,
        resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
    )

    eval_output_dir = ''
    train = (ckpt and resume) or not ckpt 
    if train:
        trainer.fit(agent, train_dataloaders=[tloader])
        eval_output_dir = Path(trainer.checkpoint_callback.best_model_path).parent

    # setup evaluator
    eval_cfg = ParamDict(vars(pargs))
    eval_cfg.update(normalizer=dataset.normalize_returns)
    Evaluator.run(agent, eval_cfg, eval_output_dir)


if __name__ == '__main__':
    main(_parse_args())
