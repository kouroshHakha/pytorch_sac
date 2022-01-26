from typing import Optional

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


########################################################
################# Goal Conditioned DT ##################
########################################################
import transformers
from decision_transformer.models.trajectory_gpt2 import GPT2Model
import einops

class GCDT(nn.Module):
    def __init__(self, conf):
        super().__init__()

        conf = utils.ParamDict(conf)
        self.conf = conf
        self.h_dim = conf.hidden_dim
        self.obs_shape = conf.obs_shape
        self.ac_dim = conf.ac_dim
        self.goal_dim = conf.goal_dim

        # embed state and actions with goals
        self.emb_state = None
        self.emb_action = None
        # embed timestep to be added to the output embs of state and action
        self.emb_timestep = None
        # the decision transformer
        self.transformer = None

        # decode state and action embs to next states and actions
        self.decode_next_state = None
        self.decode_next_action = None
        
        self._build_network()                

    def _build_network(self):
        """ builds emb_state, emb_action, emb_timestep, transformer networks
        + output decoder networks
        """

        self.emb_timestep = nn.Embedding(self.conf.max_ep_len, self.conf.hidden_dim)

        obs_shape = self.obs_shape
        h_dim = self.h_dim
        ac_dim = self.ac_dim
        g_dim = self.goal_dim

        if len(obs_shape) != 1:
            raise ValueError('obs_shape should have one dimension.')

        self.emb_state = nn.Linear(obs_shape[0] + g_dim, h_dim)
        self.emb_action = nn.Linear(ac_dim + g_dim, h_dim)

        gpt_config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=h_dim,
            n_layer=3,
            n_head=4,
            n_inner=4*h_dim,
            # activation_function='relu',
            n_positions=self.conf.max_ep_len,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            embd_pdrop=0.1,
            summary_first_dropout=0,
        )

        self.transformer = GPT2Model(gpt_config)

        self.decode_next_state = nn.Linear(h_dim, obs_shape[0])
        self.decode_next_action = nn.Linear(h_dim, ac_dim)

    def forward(self, states, actions, goal):
        """takes in state and action tokens and returns their contextualized embeddings"""
        B, T, _ = states.size()
        # simple concatenation of goal to state and action for getting their tokens

        goal_expanded = goal.unsqueeze(1).repeat(1, T, 1)
        state_in = torch.cat([states, goal_expanded], -1)
        action_in = torch.cat([actions, goal_expanded], -1)

        timesteps = torch.arange(T).to(device=states.device, dtype=torch.long)
        state_emb_in = self.emb_state(state_in) + self.emb_timestep(timesteps)
        action_emb_in = self.emb_action(action_in) + self.emb_timestep(timesteps)

        # interleave state action tokens to get 2*T
        input_tokens = einops.rearrange([state_emb_in, action_emb_in], 'i B T h -> B (T i) h')

        # attention_mask = torch.ones(input_tokens.shape[:2], dtype=torch.long, device=states.device)
        tf_outputs = self.transformer(inputs_embeds=input_tokens)
        output_tokens = tf_outputs['last_hidden_state']

        state_emb, action_emb = output_tokens[:, ::2, :], output_tokens[:, 1::2, :]

        return state_emb, action_emb

    def loss(self, pred_ac, target_ac, pred_s, target_s):

        pred_ac = pred_ac.view(-1, pred_ac.shape[-1])
        target_ac = target_ac.view(-1, target_ac.shape[-1])

        pred_s = pred_s.view(-1, pred_s.shape[-1])
        target_s = target_s.view(-1, target_s.shape[-1])

        if self.conf.loss_type == 'only_action':
            loss = F.mse_loss(pred_ac, target_ac)
        elif self.conf.loss_type == 'action_plus_dynamics':
            loss = F.mse_loss(pred_ac, target_ac) + F.mse_loss(pred_s, target_s)
        else:
            raise ValueError('invalid loss type ')

        return loss

    def ff(self, batch):
        """ compute next states and next actions """
        state, goal, act = batch
        state_embs, action_embs = self(state, act, goal)

        pred_ac = self.decode_next_action(state_embs)
        pred_ns = self.decode_next_state(action_embs)
        loss = self.loss(pred_ac, act, pred_ns, state)

        return loss, pred_ac, pred_ns

    def get_action(self, state_ctx, action_ctx, goal):
        # helper function for running the DT
        state_embs, _ = self(state_ctx, action_ctx, goal)
        pred_ac = self.decode_next_action(state_embs)
        return pred_ac

class GCDTLightningModule(GCDT, pl.LightningModule):

    def __init__(self, conf):
        super().__init__(conf)
        self.save_hyperparameters(conf)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr)

    def training_step(self, batch):
        loss, _, _ = self.ff(batch[0])
        self.log('train_loss_batch', loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([item['loss'] for item in outputs], 0)
        self.log('train_loss_epoch', losses.mean(), prog_bar=True)

########################################################
################### Trajectory BERT  ###################
########################################################
from decision_transformer.models.trajectory_bert import BertModel


class TrajBERT(nn.Module):

    def __init__(self, conf):
        super().__init__()

        conf = utils.ParamDict(conf)
        self.conf = conf
        self.h_dim = conf.hidden_dim
        self.obs_shape = conf.obs_shape
        self.ac_dim = conf.ac_dim

        self.emb_timestep = None
        self.emb_state, self.emb_action = None, None
        self.cls_token: Optional[nn.Parameter] = None
        self.transformer: Optional[BertModel] = None
        self._build_network()

    def _build_network(self):
        """ builds emb_state, emb_action, emb_timestep, transformer networks """
        self.emb_timestep = nn.Embedding(self.conf.max_ep_len, self.conf.hidden_dim)

        obs_shape = self.obs_shape
        h_dim = self.h_dim
        ac_dim = self.ac_dim

        if len(obs_shape) != 1:
            raise ValueError('obs_shape should have one dimension.')

        self.emb_state = nn.Linear(obs_shape[0], h_dim)
        self.emb_action = nn.Linear(ac_dim, h_dim)
        self.cls_token = nn.Parameter(torch.randn(h_dim), requires_grad=True)

        bert_config = transformers.BertConfig(
            vocab_size=1,
            hidden_size=h_dim,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=4*h_dim,
            max_position_embeddings=self.conf.max_ep_len,
        )

        self.transformer = BertModel(bert_config)


    def forward(self, states, actions, masks):
        # states: B, T, d_s
        # actions: B, T, d_a
        # masks: B, T (make sure it always includes first and last timesteps)

        B, T, _ = states.size()
        timesteps = torch.arange(T).to(device=states.device, dtype=torch.long)

        state_emb_in = self.emb_state(states) + self.emb_timestep(timesteps)
        action_emb_in = self.emb_action(actions) + self.emb_timestep(timesteps)

        input_tokens = einops.rearrange([state_emb_in, action_emb_in], 'i B T h -> B (T i) h')
        mask_tokens = einops.rearrange([masks, masks], 'i B T -> B (T i)')

        input_tokens = torch.cat([self.cls_token.repeat(B, 1, 1), input_tokens], dim=1)
        mask_tokens = torch.cat([torch.ones(B, 1, dtype=torch.bool).to(mask_tokens), mask_tokens], dim=1)

        tf_outputs = self.transformer(inputs_embeds=input_tokens, attention_mask=mask_tokens)
        traj_emb = tf_outputs['pooler_output']

        return traj_emb


##################################################
################### Traphormer ###################
##################################################

class TraphormerLightningModule(pl.LightningModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.save_hyperparameters(conf)

        self._build_network()

    def _build_network(self):
        enc_config = utils.ParamDict(
            hidden_dim=self.conf.hidden_dim,
            obs_shape=self.conf.obs_shape,
            ac_dim=self.conf.ac_dim,
            max_ep_len=self.conf.max_ep_len,
        )
        self.encoder = TrajBERT(enc_config)

        dec_config = utils.ParamDict(
            hidden_dim=self.conf.hidden_dim,
            max_ep_len=self.conf.max_ep_len,
            obs_shape=self.conf.obs_shape,
            ac_dim=self.conf.ac_dim,
            goal_dim=self.conf.hidden_dim,
            loss_type=self.conf.loss_type,
        )

        self.decoder = GCDT(dec_config)

    def _augment(self, state, action):
        return state, action

    def _get_mask(self, states):
        B, T, _ = states.size()
        mask_rate = self.conf.mask_rate
        # mask p percent of the input tokens for each batch
        # mask = 0 means no attention
        mask = (torch.rand(B, T) > mask_rate).long()
        # make sure the first and last timesteps are part of the input seq
        mask[:, 0] = 1
        mask[:, -1] = 1
        return mask

    def ff(self, batch):
        context_s, context_a, target_s, target_a = batch
        context_s, context_a = self._augment(context_s, context_a)
        context_mask = self._get_mask(context_s)

        task_emb = self.encoder(context_s, context_a, context_mask)
        loss, pred_ac, pred_ns = self.decoder.ff((target_s, task_emb, target_a))

        return loss, pred_ac, pred_ns

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr)

    def training_step(self, batch):
        loss, _, _ = self.ff(batch[0])
        self.log('train_loss_batch', loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([item['loss'] for item in outputs], 0)
        self.log('train_loss_epoch', losses.mean(), prog_bar=True)
