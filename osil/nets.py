
import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

import osil.utils as utils

class Encoder(nn.Module):
    def __init__(self, obs_shape, h_dim):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 1 * 35 * 35
        # self.repr_dim = h_dim

        # self.convnet = nn.Sequential(
        #     nn.Conv2d(obs_shape[0], 32, 3, stride=2),
        #     nn.ReLU(), 
        #     nn.Conv2d(32, 32, 3, stride=1),
        #     nn.ReLU(), 
        #     nn.Conv2d(32, 32, 3, stride=1),
        #     nn.ReLU(), 
        #     nn.Conv2d(32, 1, 3, stride=1),
        #     nn.ReLU()
        # )

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 4, stride=2),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, h_dim, 5, stride=1),
            nn.ReLU()
        )
        
        # self.lin = nn.Linear(self.repr_dim, h_dim, bias=True)

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h



class BC(pl.LightningModule):
    
    def __init__(self, conf):
        super().__init__()

        self.save_hyperparameters(conf)
        conf = utils.ParamDict(conf)
        self.conf = conf
        self.h_dim = conf.hidden_dim
        self.obs_shape = conf.obs_shape
        self.ac_dim = conf.ac_dim
        
        self.emb = None

        self._build_network()                
        self.apply(utils.weight_init)
        
    def forward(self, x):
        if self.emb:
            x = self.emb(x)
        pred_ac = self.mlp(x)
        return pred_ac

    def _build_network(self):
        obs_shape = self.obs_shape
        h_dim = self.h_dim
        ac_dim = self.ac_dim

        if len(obs_shape) > 1:
            self.emb = Encoder(obs_shape, h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim if self.emb else obs_shape[0], h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, ac_dim)
        )
    
    def bc_loss(self, pred_ac, target_ac):
     
        pred_ac = pred_ac.view(-1, pred_ac.shape[-1])
        target_ac = target_ac.view(-1, target_ac.shape[-1])
        loss = F.mse_loss(pred_ac, target_ac)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr)

    def ff(self, batch):
        obs, act = batch
        obs = obs.squeeze(1)
        act = act.squeeze(1)

        pred_ac = self(obs)
        loss = self.bc_loss(pred_ac, act)
        return loss, pred_ac

    def training_step(self, batch):
        loss, _ = self.ff(batch[0])
        self.log('train_loss_batch', loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([item['loss'] for item in outputs], 0)
        self.log('train_loss_epoch', losses.mean(), prog_bar=True)


class GCBC(BC):

    def __init__(self, conf):
        super().__init__(conf)

    
    def _build_network(self):
        obs_shape = self.obs_shape
        h_dim = self.h_dim
        ac_dim = self.ac_dim

        if len(obs_shape) > 1:
            self.emb = Encoder(obs_shape, h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2*h_dim if self.emb else 2*obs_shape[0], h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, ac_dim)
        )
    
    def ff(self, batch):
        obs, goal, act = batch
        obs = obs.squeeze(1)
        act = act.squeeze(1)
        goal = goal.squeeze(1)

        pred_ac = self(obs, goal)
        loss = self.bc_loss(pred_ac, act)
        return loss, pred_ac

    def forward(self, x, g):
        if self.emb:
            x = self.emb(x)
            g = self.emb(g)
        mlp_in = torch.cat([x,g], -1)
        pred_ac = self.mlp(mlp_in)
        return pred_ac