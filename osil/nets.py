from typing import Optional
from numpy import dtype

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

import osil.utils as utils

class BaseLightningModule(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self._init_conf(conf)
        self._build_network()                

    def _init_conf(self, conf):
        self.save_hyperparameters(conf)
        conf = utils.ParamDict(conf)
        self.conf = conf

    def _build_network(self):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr, weight_decay=self.conf.get('wd', 0), 
                                betas=(0.99, 0.999))

    def ff(self, batch):
        raise NotImplementedError

    def training_step(self, batch):
        loss, _ = self.ff(batch[0])
        self.log('train_loss_batch', loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([item['loss'] for item in outputs], 0)
        self.log('train_loss_epoch', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, _, = self.ff(batch)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([loss for loss in outputs], 0)
        self.log('valid_loss', losses.mean())


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
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr, weight_decay=self.conf.get('wd', 0), 
                                betas=(0.99, 0.999))

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

    def validation_step(self, batch, batch_idx):
        loss, _, = self.ff(batch)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([loss for loss in outputs], 0)
        self.log('valid_loss', losses.mean())



class GCBCv2(BaseLightningModule):

    def __init__(self, conf):
        super().__init__(conf)

    
    def _build_network(self):
        obs_dim = self.conf.obs_dim
        h_dim = self.conf.hidden_dim
        ac_dim = self.conf.ac_dim
        goal_dim = self.conf.goal_dim

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
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

    def ff(self, batch):
        x, goal, y = batch
        pred_ac = self(x, goal)
        loss = self.bc_loss(pred_ac, y)
        return loss, pred_ac

    def forward(self, x, g):
        mlp_in = torch.cat([x,g], -1)
        pred_ac = self.mlp(mlp_in)
        return pred_ac


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
            nn.Linear(2*h_dim if self.emb else obs_shape[0] + self.conf.goal_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, ac_dim)
        )
    
    def ff(self, batch):
        # obs, goal, act = batch
        # obs = obs.squeeze(1)
        # act = act.squeeze(1)

        # x, y = batch
        x, goal_vec, y = batch
        # goal_vec = goal_vec.squeeze(1)

        B, T, C = x.shape
        # goal_vec = torch.cat([x[:, :1], x[:, -1:]], -1)
        # goal_vec = torch.zeros((B, 1, 2)).to(x)
        # goal_vec = torch.cat([x[:, -1:, :2]], -1)
        goal = torch.cat([goal_vec for _ in range(T)],1)
        obs = x
        act = y

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

    def get_goal(self, state_ctx, action_ctx, start_target=0):
        
        goal_type = self.conf.goal_type
        if goal_type.startswith('last+start_qpos_qvel'):
            goal = torch.cat([state_ctx[start_target], state_ctx[-1]], -1)[None]
        elif goal_type.startswith('last+start_qpos'):
        # elif goal_type.startswith('last+start'):
            goal = torch.cat([state_ctx[start_target, :2], state_ctx[-1, :2]], -1)[None]
        elif goal_type.startswith('last_qpos'):
            goal = torch.as_tensor(state_ctx[-1][:2])[None]
        elif goal_type == 'zero':
            goal = torch.zeros(1, 2)
        else:
            raise ValueError(f'unknown goal type {goal_type}')
        return goal

    def imitate(self, env, batch_item):
        
        device = self.device
        obs, ac = batch_item
        # TODO: for now I have hard-coded the decode context length
        steps = 64

        # rst_obs should be the middle timestep
        rst_obs = obs[steps].detach().cpu().numpy()

        # random policy
        rand_trajs = get_rand_trajectories(env, rst_obs, niter=10, steps_per_episode=steps)
        
        state_ctx = torch.as_tensor(obs[:steps], dtype=torch.float, device=device)[None]
        action_ctx = torch.as_tensor(ac[:steps], dtype=torch.float, device=device)[None]
        # # goal is the xy of the last obs
        # if self.conf.goal_type == 'only_last':
        #     goal = torch.as_tensor(obs[-1][:2], dtype=torch.float, device=device)[None]
        # elif self.conf.goal_type == 'zero':
        #     goal = torch.zeros(2).float().to(device)[None]
        # elif self.conf.goal_type == 'last+start':
        #     goal = torch.cat([obs[steps, :2], obs[-1, :2]], -1).float().to(device)[None]
        #     # goal = torch.cat([obs[steps], obs[-1]], -1).float().to(device)[None]

        output = dict(rand_trajs=rand_trajs)
        for start in [0, steps]:
            # run two goal embeddings: 1) start is where episode len becomes 64 2) start is the beginning of the ep (e.g. ep_len=128)
            goal = self.get_goal(obs, ac, start_target=start).to(device)

            env.reset()
            rst_obs = obs[steps].detach().cpu().numpy()
            env.set_state(rst_obs[:2], rst_obs[2:])
            traj = {'states': [o.detach().cpu().numpy() for o in obs[:steps]], 'actions': [a.detach().cpu().numpy() for a in ac[:steps]]}
            for t in range(len(obs) - steps):
                a = self(state_ctx[:, -1], goal).squeeze(0)
                a = a.detach().cpu().numpy()
                s, _, _, _ = env.step(a)

                traj['states'].append(s)
                traj['actions'].append(a)

                # convert numpy to batched tensors (1, 1, D)
                s = torch.as_tensor(s, dtype=torch.float, device=device)[None, None]
                a = torch.as_tensor(a, dtype=torch.float, device=device)[None, None]

                # context is bootstraped on policies' own actions
                state_ctx = torch.cat([state_ctx[:, 1:], s], 1)
                action_ctx = torch.cat([action_ctx[:, 1:], a], 1)

            output[f'policy_traj_{start}'] = traj

        return output

########################################################
################# Goal Conditioned DT ##################
########################################################
import transformers
from decision_transformer.models.trajectory_gpt2 import GPT2Model
import einops


def get_rand_trajectories(env, rst_obs, niter, steps_per_episode):
    rand_trajs = []
    for _ in range(niter):
        env.reset()
        env.set_state(rst_obs[:2], rst_obs[2:])

        traj = {'states': [], 'actions': []}
        for _ in range(steps_per_episode):
            a = env.action_space.sample()
            s, _, _, _  = env.step(a)
            traj['states'].append(s)
            traj['actions'].append(a)
        rand_trajs.append(traj)      
    return rand_trajs         


class GCDT(nn.Module):
    def __init__(self, conf):
        super().__init__()

        conf = utils.ParamDict(conf)
        self.conf = conf
        self.h_dim = conf.hidden_dim
        self.obs_shape = conf.obs_shape
        self.ac_dim = conf.ac_dim
        self.goal_dim = conf.goal_dim
        self.decode_size = conf.decoder_size

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
        # TODO
        input_tokens = einops.rearrange([state_emb_in, action_emb_in], 'i B T h -> B (T i) h')
        # input_tokens = state_emb_in

        # attention_mask = torch.ones(input_tokens.shape[:2], dtype=torch.long, device=states.device)
        tf_outputs = self.transformer(inputs_embeds=input_tokens)
        output_tokens = tf_outputs['last_hidden_state']

        state_emb, action_emb = output_tokens[:, ::2, :], output_tokens[:, 1::2, :]
        # state_emb, action_emb = output_tokens, None

        return state_emb, action_emb

    def loss(self, pred_ac, target_ac, pred_s, target_s, loss_type):
        pred_ac_flat = pred_ac.view(-1, pred_ac.shape[-1])
        target_ac_flat = target_ac.view(-1, target_ac.shape[-1])

        pred_s_flat = pred_s.view(-1, pred_s.shape[-1])
        target_s_flat = target_s.view(-1, target_s.shape[-1])

        if loss_type == 'only_action':
            loss = F.mse_loss(pred_ac_flat, target_ac_flat)
        elif loss_type == 'only_last_action':
            loss = F.mse_loss(pred_ac[:, -1], target_ac[:, -1])
        elif loss_type == 'action_plus_dynamics':
            loss = F.mse_loss(pred_ac_flat, target_ac_flat) + F.mse_loss(pred_s_flat, target_s_flat)
        else:
            raise ValueError('invalid loss type ')

        return loss

    def ff(self, batch, loss_type):
        """ compute next states and next actions """
        state, goal, act = batch
        state_embs, action_embs = self(state, act, goal)

        pred_ac = self.decode_next_action(state_embs)
        # TODO
        pred_ns = self.decode_next_state(action_embs)
        # pred_ns = state
        loss = self.loss(pred_ac, act, pred_ns, state, loss_type)

        return loss, pred_ac, pred_ns

    def get_action(self, state_ctx, action_ctx, goal):
        # helper function for running the DT
        state_embs, _ = self(state_ctx, action_ctx, goal)
        pred_ac = self.decode_next_action(state_embs)
        # batched but keep the last step
        return pred_ac[:, -1]

    def imitate(self, env, batch_item):
        
        device = self.transformer.device
        obs, ac = batch_item
        # TODO: for now I have hard-coded the decode context length
        steps = self.decode_size

        # rst_obs should be the middle timestep
        rst_obs = obs[steps].detach().cpu().numpy()

        # random policy
        rand_trajs = get_rand_trajectories(env, rst_obs, niter=10, steps_per_episode=steps)
        
        state_ctx = torch.as_tensor(obs[:steps], dtype=torch.float, device=device)[None]
        action_ctx = torch.as_tensor(ac[:steps], dtype=torch.float, device=device)[None]
        # goal is the xy of the last obs
        if self.conf.goal_type == 'only_last':
            goal = torch.as_tensor(obs[-1][:2], dtype=torch.float, device=device)[None]
        elif self.conf.goal_type == 'zero':
            goal = torch.zeros(2).float().to(device)[None]
        elif self.conf.goal_type == 'last+start':
            goal = torch.cat([obs[0, :2], obs[-1, :2]], -1).float().to(device)[None]

        env.reset()
        env.set_state(rst_obs[:2], rst_obs[2:])
        traj = {'states': [o.detach().cpu().numpy() for o in obs[:steps]], 'actions': [a.detach().cpu().numpy() for a in ac[:steps]]}
        for t in range(len(obs) - steps):
            a = self.get_action(state_ctx, action_ctx, goal).squeeze(0)
            a = a.detach().cpu().numpy()
            s, _, _, _ = env.step(a)

            traj['states'].append(s)
            traj['actions'].append(a)

            # convert numpy to batched tensors (1, 1, D)
            s = torch.as_tensor(s, dtype=torch.float, device=device)[None, None]
            a = torch.as_tensor(a, dtype=torch.float, device=device)[None, None]

            # context is bootstraped on policies' own actions
            state_ctx = torch.cat([state_ctx[:, 1:], s], 1)
            action_ctx = torch.cat([action_ctx[:, 1:], a], 1)

        output = dict(
            rand_trajs=rand_trajs,
            policy_traj=traj
        )

        return output

class GCDTLightningModule(GCDT, pl.LightningModule):

    def __init__(self, conf):
        super().__init__(conf)
        self.save_hyperparameters(conf)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr)

    def training_step(self, batch):
        loss, _, _ = self.ff(batch[0], loss_type=self.conf.loss_type)
        self.log('train_loss_batch', loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([item['loss'] for item in outputs], 0)
        self.log('train_loss_epoch', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # loss, _, _ = self.ff(batch, loss_type='only_last_action')
        loss, _, _ = self.ff(batch, loss_type='only_action')
        return loss

    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([loss for loss in outputs], 0)
        self.log('valid_loss', losses.mean())

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
            num_attention_heads=8,
            intermediate_size=h_dim,
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
        self.conf = utils.ParamDict(conf)
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
        mask = (torch.rand(B, T) > mask_rate).long().to(device=states.device)
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

    def get_goal(self, state_ctx, action_ctx, mask=True, augment=True):

        if augment:
            state_ctx, action_ctx = self._augment(state_ctx, action_ctx)

        if mask:
            mask_ctx = self._get_mask(state_ctx)
        else:
            mask_ctx = torch.ones(state_ctx.shape[:2], dtype=torch.long, device=state_ctx.device)

        task_emb = self.encoder(state_ctx, action_ctx, mask_ctx)
        return task_emb

    def imitate(self, env, batch_item):
        
        device = self.device
        obs, ac = batch_item
        steps = self.decoder.decode_size

        # rst_obs should be the middle timestep
        rst_obs = obs[steps].detach().cpu().numpy()

        # random policy
        rand_trajs = get_rand_trajectories(env, rst_obs, niter=10, steps_per_episode=steps)
        
        state_ctx = torch.as_tensor(obs[:steps], dtype=torch.float, device=device)[None]
        action_ctx = torch.as_tensor(ac[:steps], dtype=torch.float, device=device)[None]

        # TODO: What should goal embedding be during inference? 
        # should goal be encoding of c_s, and c_a or obs, ac? with or without mask?
        goal = self.get_goal(obs[None].to(device), ac[None].to(device), mask=True, augment=True)

        env.reset()
        env.set_state(rst_obs[:2], rst_obs[2:])
        traj = {'states': [o.detach().cpu().numpy() for o in obs[:steps]], 'actions': [a.detach().cpu().numpy() for a in ac[:steps]]}
        for t in range(steps):
            a = self.decoder.get_action(state_ctx, action_ctx, goal).squeeze(0)
            # # copying open loop actions
            # a = ac[steps + t]
            a = a.detach().cpu().numpy()
            s, _, _, _ = env.step(a)

            traj['states'].append(s)
            traj['actions'].append(a)

            # convert numpy to batched tensors (1, 1, D)
            s = torch.as_tensor(s, dtype=torch.float, device=device)[None, None]
            a = torch.as_tensor(a, dtype=torch.float, device=device)[None, None]

            # context is bootstraped on policies' own actions
            state_ctx = torch.cat([state_ctx[:, 1:], s], 1)
            action_ctx = torch.cat([action_ctx[:, 1:], a], 1)

        output = dict(
            rand_trajs=rand_trajs,
            policy_traj=traj
        )

        return output

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr)

    def training_step(self, batch):
        loss, _, _ = self.ff(batch[0])
        self.log('train_loss_batch', loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([item['loss'] for item in outputs], 0)
        self.log('train_loss_epoch', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.ff(batch)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([loss for loss in outputs], 0)
        self.log('valid_loss', losses.mean())



###################################################################
################### Traphormer with MLP decoder ###################
###################################################################

class TraphormerBCLightningModule(pl.LightningModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = utils.ParamDict(conf)
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
        self.goal_dim = self.conf.goal_dim
        self.goal_emb = nn.Linear(self.conf.hidden_dim, self.goal_dim)

        dec_config = utils.ParamDict(
            hidden_dim=self.conf.hidden_dim,
            obs_shape=self.conf.obs_shape,
            ac_dim=self.conf.ac_dim,
            goal_dim=self.conf.goal_dim,
            # loss_type=self.conf.loss_type,
            goal_type=self.conf.goal_type,
        )

        self.decoder = GCBC(dec_config)

    def _augment(self, state, action):
        return state, action

    def _get_mask(self, states):
        B, T, _ = states.size()
        mask_rate = self.conf.mask_rate
        # mask p percent of the input tokens for each batch
        # mask = 0 means no attention
        mask = (torch.rand(B, T) > mask_rate).long().to(device=states.device)
        # make sure the first and last timesteps are part of the input seq
        mask[:, 0] = 1
        mask[:, -1] = 1
        return mask

    def ff(self, batch):
        context_s, context_a, target_s, target_a = batch
        context_s, context_a = self._augment(context_s, context_a)
        context_mask = self._get_mask(context_s)

        task_emb = self.encoder(context_s, context_a, context_mask)
        task_emb = self.goal_emb(task_emb)
        loss, pred_ac = self.decoder.ff((target_s, task_emb.unsqueeze(1), target_a))

        return loss, pred_ac

    def get_goal(self, state_ctx, action_ctx, mask=True, augment=True):

        if augment:
            state_ctx, action_ctx = self._augment(state_ctx, action_ctx)

        if mask:
            mask_ctx = self._get_mask(state_ctx)
        else:
            mask_ctx = torch.ones(state_ctx.shape[:2], dtype=torch.long, device=state_ctx.device)

        task_emb = self.encoder(state_ctx, action_ctx, mask_ctx)
        task_emb = self.goal_emb(task_emb)
        return task_emb

    def imitate(self, env, batch_item):
        
        device = self.device
        obs, ac = batch_item
        steps = 64

        # rst_obs should be the middle timestep
        rst_obs = obs[steps].detach().cpu().numpy()

        # random policy
        rand_trajs = get_rand_trajectories(env, rst_obs, niter=10, steps_per_episode=steps)
        
        state_ctx = torch.as_tensor(obs[:steps], dtype=torch.float, device=device)[None]
        action_ctx = torch.as_tensor(ac[:steps], dtype=torch.float, device=device)[None]

        output = dict(rand_trajs=rand_trajs)
        for mask in [False, True]:
            # TODO: What should goal embedding be during inference? 
            # should goal be encoding of c_s, and c_a or obs, ac? with or without mask?
            goal = self.get_goal(obs[None].to(device), ac[None].to(device), mask=mask, augment=True)

            env.reset()
            env.set_state(rst_obs[:2], rst_obs[2:])
            traj = {'states': [o.detach().cpu().numpy() for o in obs[:steps]], 'actions': [a.detach().cpu().numpy() for a in ac[:steps]]}
            for t in range(steps):
                a = self.decoder(state_ctx[:, -1], goal).squeeze(0)
                # # copying open loop actions
                # a = ac[steps + t]
                a = a.detach().cpu().numpy()
                s, _, _, _ = env.step(a)

                traj['states'].append(s)
                traj['actions'].append(a)

                # convert numpy to batched tensors (1, 1, D)
                s = torch.as_tensor(s, dtype=torch.float, device=device)[None, None]
                a = torch.as_tensor(a, dtype=torch.float, device=device)[None, None]

                # context is bootstraped on policies' own actions
                state_ctx = torch.cat([state_ctx[:, 1:], s], 1)
                action_ctx = torch.cat([action_ctx[:, 1:], a], 1)

                output[f'policy_traj_mask={mask}'] = traj

        return output

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr, weight_decay=self.conf.wd, betas=(0.99, 0.999))

    def training_step(self, batch):
        loss, _ = self.ff(batch[0])
        self.log('train_loss_batch', loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([item['loss'] for item in outputs], 0)
        self.log('train_loss_epoch', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, _ = self.ff(batch)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([loss for loss in outputs], 0)
        self.log('valid_loss', losses.mean())
