"""
Script for Pre-trained/Fine-tuned behavioral cloning
"""

import argparse
from pathlib import Path
from torch.utils.data.dataloader import DataLoader

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from osil.data import GCDMC, Concat
from osil.nets import GCBC
from osil.utils import ParamDict
from osil.eval import EvaluatorGC

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

    return parser.parse_args()


def main(pargs):

    domain = pargs.task_name.split('_')[0]
    if domain == 'walker':
        tasks = ['walk', 'run', 'stand', 'flip']
    elif domain == 'quadruped':
        tasks = ['walk', 'run', 'stand', 'jump']
    else:
        raise ValueError(f'Domain {domain} not recognized!')
    
    ########################################################################
    #  pre-training a gc bc agent on the entire dataset we have from the domain
    ########################################################################
    exp_name = f'pt_gcbc_{domain}_{pargs.obs_type}'
    print(f'Running {exp_name} ...')
    
    print('Loading datasets ...')
    datasets = []
    ft_dset = None
    for task in tasks:
        dataset = GCDMC(f'{domain}_{task}', pargs.obs_type, block_size=1)
        datasets.append(dataset)
        if task in pargs.task_name:
            ft_dset = dataset
    print('Datasets loaded.')

    dset = Concat(datasets)
    tloader = DataLoader(dset, shuffle=True, batch_size=pargs.batch_size, num_workers=4)

    obs, _, act = dset[0]
    
    configs = ParamDict(vars(pargs))
    ckpt = configs.pop('ckpt', '')
    resume = configs.pop('resume', False)

    bc_config = ParamDict(configs.copy())
    bc_config.update(obs_shape=obs.shape[1:], ac_dim=act.shape[-1])
    
    agent = GCBC.load_from_checkpoint(ckpt) if ckpt else GCBC(bc_config)
    logger = TensorBoardLogger(save_dir='tb_logs', name=exp_name)
    trainer = pl.Trainer(
        max_epochs=pargs.max_epochs,
        resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
    )

    eval_output_dir = ''
    pretrain = (ckpt and resume) or not ckpt 
    if pretrain:
        trainer.fit(agent, train_dataloaders=[tloader])
        eval_output_dir = Path(trainer.checkpoint_callback.best_model_path).parent

    # setup evaluator for zero-shot adaptation by running goal conditioned policy
    print(f'Evaluating pretrained agent on {pargs.task_name}...')
    eval_cfg = ParamDict(configs.copy())
    eval_cfg.update(normalizer=ft_dset.normalize_returns, ft_dset=ft_dset)

    if not eval_output_dir:
        eval_output_dir = Path(ckpt).parent
    eval_output_dir = eval_output_dir / f'eval_{pargs.task_name}'
    eval_output_dir.mkdir(exist_ok=True)
    
    EvaluatorGC(agent, eval_cfg, eval_output_dir).run()
    print('Evaluating pretrained goal conditioned agent done.')



if __name__ == '__main__':
    main(_parse_args())
