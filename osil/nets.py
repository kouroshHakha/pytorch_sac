from base64 import encode
from collections import defaultdict
from typing import Optional
from numpy import dtype, inner

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

import dataclasses

import osil.utils as utils
from osil.resnet import ResNetBasicBlock



class MLP(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_dim, n_layers, activation=nn.ReLU(), bnorm=False, input_norm=None) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation

        self.net = []

        dims = [in_channel] + [hidden_dim] * n_layers + [out_channel]
        net = []
        if input_norm:
            net.append(input_norm)
        for h1, h2 in zip(dims[:-1], dims[1:]):
            net.append(nn.Linear(h1, h2))
            if bnorm:
                net.append(nn.BatchNorm1d(h2))
            net.append(activation)
        net.pop()

        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)

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

    def ff(self, batch, compute_loss=True):
        raise NotImplementedError

    def training_step(self, batch):
        ret = self.ff(batch[0], compute_loss=True)
        self.log('train_loss_batch', ret['loss'])
        return ret['loss']

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([item['loss'] for item in outputs], 0)
        self.log('train_loss_epoch', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        ret = self.ff(batch, compute_loss=True)
        return ret['loss']

    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([loss for loss in outputs], 0)
        self.log('valid_loss', losses.mean(), prog_bar=True)


class Encoder(nn.Module):
    def __init__(self, obs_shape, h_dim):
        super().__init__()

        assert len(obs_shape) == 3
        assert obs_shape[1] == obs_shape[2] == 64

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2), # 31x31
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=2), # 15x15
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=2), # 7x7
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), # 3x3
            nn.ReLU(),
            nn.Conv2d(32, h_dim, 3, stride=2), # 1x1
            nn.ReLU(),
        )
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Encoderv2(nn.Module):
    def __init__(self, obs_shape, h_dim):
        super().__init__()

        assert len(obs_shape) == 3
        assert obs_shape[1] == obs_shape[2] == 64

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 5, stride=2, padding=2), # 32x32
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=4, stride=4), # 8x8
            #############
            nn.Conv2d(32, 32, 3, stride=1, padding='same'), # 8x8
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), # 4x4
            #############
            nn.Conv2d(32, 32, 3, stride=1, padding='same'), # 4x4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2x2
            #############
            nn.Conv2d(32, h_dim, 3, stride=1, padding='same'), # 2x2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1x1
        )
        
                
        x = torch.rand(obs_shape)[None]
        y = self.convnet(x)
        assert y.shape[1:] == (h_dim, 1, 1)
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class EncoderWithBnorm(Encoder):

    def __init__(self, obs_shape, h_dim):
        super().__init__(obs_shape, h_dim)

        assert len(obs_shape) == 3
        assert obs_shape[1] == obs_shape[2] == 64

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2), # 31x31
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=2), # 15x15
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=2), # 7x7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), # 3x3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, h_dim, 3, stride=2), # 1x1
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
        )
        
        self.apply(utils.weight_init)

class EncoderResNet(Encoder):

    def __init__(self, obs_shape, h_dim):
        super().__init__(obs_shape, h_dim)

        assert len(obs_shape) == 3
        assert obs_shape[1] == obs_shape[2] == 64

        self.convnet = nn.Sequential(
            ResNetBasicBlock(3, 32, downsampling=2),
            ResNetBasicBlock(32, 32, downsampling=2),
            ResNetBasicBlock(32, 32, downsampling=2),
            ResNetBasicBlock(32, 32, downsampling=2),
            ResNetBasicBlock(32, 32, downsampling=2),
            ResNetBasicBlock(32, h_dim, downsampling=2),
        )


class Encoder28(nn.Module):
    def __init__(self, obs_shape, h_dim):
        super().__init__()

        assert len(obs_shape) == 3
        assert obs_shape[1] == obs_shape[2] == 28

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=1, padding=1), # 32x28x28
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x14x14
            nn.Conv2d(32, 32, 3, stride=1, padding=1), # 32x14x14
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x7x7
            nn.Conv2d(32, 64, 3, stride=1), # 64x4x4
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x2x2
        )
        
        # x = torch.rand(obs_shape)[None]
        # y = self.convnet(x).flatten()
        # self.linear = nn.Linear(y.shape[0], h_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        # h = self.linear(h)
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
        self.log('valid_loss', losses.mean(), prog_bar=True)



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

    def ff(self, batch, compute_loss=True):
        x, goal, y = batch
        pred_ac = self(x, goal)
        ret = dict(pred_ac=pred_ac)
        if compute_loss:
            loss = self.bc_loss(pred_ac, y)
            ret['loss']=loss
        return ret

    def forward(self, x, g):
        assert x.shape[-1] == self.conf.obs_dim, 'state shape is not correct.'
        assert g.shape[-1] == self.conf.goal_dim, 'goal shape is not correct.'
        mlp_in = torch.cat([x,g], -1)
        pred_ac = self.mlp(mlp_in)
        # TODO: limiting actions to -1, 1
        pred_ac = pred_ac.tanh()
        return pred_ac


class GCBCv3(BaseLightningModule):

    def __init__(self, conf):
        super().__init__(conf)

    def _check_obs_and_goal(self, obs, goal):
        assert obs.shape[1:] == self.conf.obs_shape, 'observation shape is not correct.'
        assert goal.shape[1:] == self.conf.goal_shape, 'goal shape is not correct.'

    
    def _build_network(self):
        self.is_obs_image = len(self.conf.obs_shape) > 1
        self.is_goal_image = len(self.conf.goal_shape) > 1
        h_dim = self.conf.hidden_dim
        ac_dim = self.conf.ac_dim
        self.image_enc_hdim = image_enc_hdim = self.conf.goal_enc_dim

        self.enc_obs = None
        self.enc_goal = None
        self.enc = None
        if self.is_obs_image:
            obs_shape = self.conf.obs_shape
            goal_shape = self.conf.goal_shape

            # trying out 28x28 input
            encoder_cls = EncoderResNet if obs_shape[-1] == 64 else Encoder28
            # encoder_cls = Encoder if obs_shape[-1] == 64 else Encoder28

            if self.is_goal_image:
                assert obs_shape[-1] == goal_shape[-1], 'Observation and goal images should be of the same W'
                assert obs_shape[-2] == goal_shape[-2], 'Observation and goal images should be of the same H'
                # enc_in_shape = (obs_shape[0] + goal_shape[0], ) + tuple(obs_shape[1:])

                # enc_in_shape = (obs_shape[0] ,) + tuple(obs_shape[1:])
                enc_in_shape = (goal_shape[0] ,) + tuple(obs_shape[1:])
                self.enc = encoder_cls(enc_in_shape, image_enc_hdim)
                # self.enc_obs = encoder_cls(obs_shape, image_enc_hdim)
                # self.enc_goal = encoder_cls(goal_shape, image_enc_hdim)
                
                # mlp_channel_in = h_dim 
                # mlp_channel_in = h_dim*2
                self.n_stack = obs_shape[0] // goal_shape[0]
                mlp_channel_in = (1 + self.n_stack) * image_enc_hdim
            else:
                self.enc_obs = encoder_cls(obs_shape, image_enc_hdim)
                # self.enc = encoder_cls(obs_shape, h_dim)
                mlp_channel_in = h_dim + self.conf.goal_shape[-1]
        else:
            mlp_channel_in = self.conf.obs_shape[-1] + self.conf.goal_shape[-1]

        self.mlp = MLP(mlp_channel_in, ac_dim, h_dim, n_layers=4, activation=nn.ReLU())
        self.color_network = MLP(image_enc_hdim, 3, h_dim, 3)

    
    def bc_loss(self, pred_ac, target_ac):
        pred_ac = pred_ac.view(-1, pred_ac.shape[-1])
        target_ac = target_ac.view(-1, target_ac.shape[-1])
        loss = F.mse_loss(pred_ac, target_ac)
        
        return loss

    def ff(self, batch, compute_loss=True):
        obs, goal, act, target_color = batch
        pred_ac, pred_target_color = self(obs, goal)
        ret = dict(pred_ac=pred_ac.detach())
        if compute_loss:
            loss = self.bc_loss(pred_ac, act)
            loss_color = self.bc_loss(pred_target_color, target_color)
            ret['loss'] = loss + loss_color
            ret['loss_action'] = loss.detach()
            ret['loss_color'] = loss_color.detach()
        return ret

    def forward(self, obs, goal):

        self._check_obs_and_goal(obs, goal)

        if self.is_obs_image and self.is_goal_image:
            # enc_in = torch.cat([obs, goal], dim=1) # cat dim = C
            # mlp_in = self.enc(enc_in)
            # mlp_in = torch.cat([self.enc_obs(obs), self.enc_goal(goal)], -1)

            # mlp_in = torch.cat([self.enc(obs), self.enc(goal)], -1)
            obses = [obs[:, i*3:i*3+3] for i in range(self.n_stack)]
            encodings = [self.enc(x) for x in obses + [goal]]
            mlp_in = torch.cat(encodings, -1)
            pred_target_color = self.color_network(encodings[-1])
        else:
            assert not(self.is_goal_image and not self.is_obs_image) # cannot have goal img but obs state
            
            x = self.enc_obs(obs) if self.is_obs_image else obs
            # g = self.enc(goal) if self.is_goal_image else goal
            mlp_in = torch.cat([x, goal], -1)

        pred_ac = self.mlp(mlp_in)
        # TODO: limiting actions to -1, 1
        pred_ac = pred_ac.tanh()
        return pred_ac, pred_target_color
        # return pred_ac

    def training_step(self, batch):
        ret = self.ff(batch[0], compute_loss=True)
        self.log('train_loss_batch', ret['loss_action'])
        self.log('train_loss_total', ret['loss'])
        self.log('train_loss_color', ret['loss_color'])
        return ret

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([item['loss_action'] for item in outputs], 0)
        self.log('train_loss_epoch', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        ret = self.ff(batch, compute_loss=True)
        return ret

    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([output['loss_action'] for output in outputs], 0)
        color_losses = torch.stack([output['loss_color'] for output in outputs], 0)
        self.log('valid_loss', losses.mean(), prog_bar=True)
        self.log('valid_loss_color', color_losses.mean(), prog_bar=True)


class GCBCv4(GCBCv3):


    def _build_network(self):
        self.is_obs_image = len(self.conf.obs_shape) > 1
        self.is_goal_image = len(self.conf.goal_shape) > 1
        h_dim = self.conf.hidden_dim
        ac_dim = self.conf.ac_dim


        self.image_enc_hdim = image_enc_hdim = h_dim #self.conf.goal_enc_dim

        self.enc_obs = None
        self.enc_goal = None
        self.enc = None
        if self.is_obs_image:
            obs_shape = self.conf.obs_shape
            goal_shape = self.conf.goal_shape


            # trying out 28x28 input
            encoder_cls = EncoderResNet if obs_shape[-1] == 64 else Encoder28
            # encoder_cls = Encoder if obs_shape[-1] == 64 else Encoder28

            if self.is_goal_image:
                assert obs_shape[-1] == goal_shape[-1], 'Observation and goal images should be of the same W'
                assert obs_shape[-2] == goal_shape[-2], 'Observation and goal images should be of the same H'
                enc_in_shape = (goal_shape[0] ,) + tuple(obs_shape[1:])
                self.enc = encoder_cls(enc_in_shape, image_enc_hdim)
                self.n_stack = obs_shape[0] // goal_shape[0]
                # mlp_channel_in = (1 + self.n_stack) * image_enc_hdim
                mlp_channel_in = 2 + self.n_stack * image_enc_hdim
        #     else:
        #         self.enc_obs = encoder_cls(obs_shape, image_enc_hdim)
        #         # self.enc = encoder_cls(obs_shape, h_dim)
        #         mlp_channel_in = h_dim + self.conf.goal_shape[-1]
        # else:
        #     mlp_channel_in = self.conf.obs_shape[-1] + self.conf.goal_shape[-1]

        # max_token_len = self.n_stack + 1
        # self.emb_timestep = nn.Embedding(max_token_len, h_dim) # nstack + 1
        # gpt_config = transformers.GPT2Config(
        #     vocab_size=1,
        #     n_embd=h_dim,
        #     n_layer=3,
        #     n_head=4,
        #     n_inner=h_dim,
        #     # reserve one extra token for cls which will be added in the model
        #     n_positions=max_token_len,
        # )

        # self.token_in = nn.Linear(image_enc_hdim, h_dim)
        # self.gpt = GPT2Model(gpt_config)
        # self.decode_action = MLP(h_dim, ac_dim, h_dim, 3)

        self.mlp = MLP(mlp_channel_in, ac_dim, h_dim, 3)
        self.eef_network = MLP(2 * self.image_enc_hdim, 2, self.conf.hidden_dim, 3)

        if self.conf.get('use_contrastive', False):
            self.contra_proj = MLP(image_enc_hdim, h_dim, h_dim, n_layers=1)

        self.color_network = MLP(image_enc_hdim, 3, h_dim, 3)

    def ff(self, batch, compute_loss=True):
        obs, goal, act, target_eef, goal_aug = batch
        pred_ac, pred_eef = self(obs, goal)
        ret = dict(pred_ac=pred_ac.detach())
        if compute_loss:
            loss = self.bc_loss(pred_ac, act)
            loss_eef = self.bc_loss(pred_eef, target_eef)
            ret['loss'] = loss + loss_eef 
            if self.conf.get('use_contrastive', False):
                loss_contra = self.contra_loss(goal, goal_aug)
                ret['loss_contra'] = loss_contra.detach()
                ret['loss'] += 0.1 * loss_contra

            ret['loss_action'] = loss.detach()
            ret['loss_eef'] = loss_eef.detach()

        # if not self.training:

        #     def show_batch(idx):
        #         import matplotlib.pyplot as plt
        #         plt.subplot(131)
        #         plt.imshow(obs[idx, -3:].detach().cpu().permute(1, 2, 0))
        #         plt.subplot(132)
        #         plt.imshow(goal[idx].detach().cpu().permute(1, 2, 0))
        #         plt.subplot(133)
        #         plt.imshow(goal_aug[idx].detach().cpu().permute(1, 2, 0))
        #         plt.savefig('in_training_eef.png')
        #         print(pred_ac[idx])
        #         print(pred_eef[idx])

        #     show_batch(0)
        #     breakpoint()

        return ret

    def contra_loss(self, goal, goal_aug):
        x = self.enc(goal)
        x_aug = self.enc(goal_aug)

        z = self.contra_proj(x)
        z_aug = self.contra_proj(x_aug)
        loss = ContrastiveLossELI5(x.shape[0])(z, z_aug)

        return loss

    def forward(self, obs, goal):

        self._check_obs_and_goal(obs, goal)

        if self.is_obs_image and self.is_goal_image:
            # enc_in = torch.cat([obs, goal], dim=1) # cat dim = C
            # mlp_in = self.enc(enc_in)
            # mlp_in = torch.cat([self.enc_obs(obs), self.enc_goal(goal)], -1)

            # mlp_in = torch.cat([self.enc(obs), self.enc(goal)], -1)
            obses = [obs[:, i*3:i*3+3] for i in range(self.n_stack)]
            encodings = [self.enc(x) for x in obses + [goal]]
            # catenate current obs and goal encodings
            pred_eef = self.eef_network(torch.cat([encodings[-2], encodings[-1]], -1))
            pred_eef = pred_eef.tanh()
        # else:
        #     assert not(self.is_goal_image and not self.is_obs_image) # cannot have goal img but obs state
            
        #     x = self.enc_obs(obs) if self.is_obs_image else obs
        #     # g = self.enc(goal) if self.is_goal_image else goal
        #     mlp_in = torch.cat([x, goal], -1)

        # mlp_in = torch.cat(encodings, -1)
        mlp_in = torch.cat(encodings[:-1] + [pred_eef], -1)
        pred_ac = self.mlp(mlp_in)
        # TODO: limiting actions to -1, 1

        # # contextualize the encodings
        # encodings = torch.stack(encodings, 1) # B, T, h
        # B, T, _ = encodings.size()
        # timesteps = torch.arange(T).to(device=encodings.device, dtype=torch.long)
        # encodings_ordered = self.token_in(encodings) + self.emb_timestep(timesteps)
        # tf_output = self.gpt(inputs_embeds=encodings_ordered, output_attentions=True)
        # pred_ac = self.decode_action(tf_output['last_hidden_state'][:, -1])

        pred_ac = pred_ac.tanh()

        return pred_ac, pred_eef

    def training_step(self, batch):
        ret = self.ff(batch[0], compute_loss=True)
        self.log('train_loss_batch', ret['loss_action'])
        self.log('train_loss_total', ret['loss'])
        self.log('train_loss_eef', ret['loss_eef'])
        # self.log('train_loss_contra', ret['loss_contra'])
        return ret


    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([output['loss_action'] for output in outputs], 0)
        eef_losses = torch.stack([output['loss_eef'] for output in outputs], 0)
        # contra_losses = torch.stack([output['loss_contra'] for output in outputs], 0)
        self.log('valid_loss', losses.mean(), prog_bar=True)
        self.log('valid_loss_eef', eef_losses.mean(), prog_bar=True)
        # self.log('valid_loss_contra', contra_losses.mean(), prog_bar=True)


class GCBCv5(BaseLightningModule):

    enc_cls = {
        'normal': Encoder,
        'resnet': EncoderResNet,
        'normal_bnorm': EncoderWithBnorm,
    }
    def __init__(self, conf):
        super().__init__(conf)

    def _check_obs_and_goal(self, obs, goal):
        assert obs.shape[1:] == self.conf.obs_shape, 'observation shape is not correct.'
        assert goal.shape[1:] == self.conf.goal_shape, 'goal shape is not correct.'

    def _build_network(self):
        self.is_obs_image = len(self.conf.obs_shape) > 1
        self.is_goal_image = len(self.conf.goal_shape) > 1
        h_dim = self.conf.hidden_dim
        ac_dim = self.conf.ac_dim

        ctrl_net_nlayers = self.conf.get('ctrl_net_nlayers', 3)
        self.image_enc_hdim = image_enc_hdim = self.conf.goal_enc_dim

        self.enc = None
        if self.is_obs_image:
            obs_shape = self.conf.obs_shape
            goal_shape = self.conf.goal_shape
            enc_type = self.conf.get('enc_type', 'normal')
            enc_cls = self.enc_cls[enc_type]

            if self.is_goal_image:
                assert obs_shape[-1] == goal_shape[-1], 'Observation and goal images should be of the same W'
                assert obs_shape[-2] == goal_shape[-2], 'Observation and goal images should be of the same H'
                enc_in_shape = (goal_shape[0] ,) + tuple(obs_shape[1:])
                self.enc = enc_cls(enc_in_shape, image_enc_hdim)
                self.n_stack = obs_shape[0] // goal_shape[0]
                mlp_channel_in = (1 + self.n_stack) * image_enc_hdim
            else:
                self.enc = enc_cls(obs_shape, image_enc_hdim)
                mlp_channel_in = h_dim + self.conf.goal_shape[-1]
        else:
            mlp_channel_in = self.conf.obs_shape[-1] + self.conf.goal_shape[-1]

        self.ctrl_net = MLP(mlp_channel_in, ac_dim, h_dim, ctrl_net_nlayers)

        if self.conf.get('use_target_color_loss', False):
            # color_dim = 3
            self.color_net = MLP(image_enc_hdim, 3, h_dim, 3)

        if self.conf.get('use_target_eef_loss', False):
            # eef_dim = 2
            self.eef_net = MLP(2 * image_enc_hdim, 2, h_dim, 3)

        if self.conf.get('use_contrastive_loss', False):
            # projection head
            self.contra_proj = MLP(image_enc_hdim, h_dim, h_dim, n_layers=1)
        

    def bc_loss(self, pred_ac, target_ac):
        pred_ac = pred_ac.view(-1, pred_ac.shape[-1])
        target_ac = target_ac.view(-1, target_ac.shape[-1])

        if self.conf.get('use_huber_loss', False):
            loss = nn.HuberLoss(delta=0.05)(pred_ac, target_ac)
        else:
            loss = nn.MSELoss()(pred_ac, target_ac)
        
        return loss

    def contra_loss(self, x, x_aug):

        z = self.contra_proj(x)
        z_aug = self.contra_proj(x_aug)
        loss = ContrastiveLossELI5(x.shape[0])(z, z_aug)

        return loss

    def ff(self, batch, compute_loss=True):
        act = batch['action']

        use_contrastive_loss = self.conf.get('use_contrastive_loss', False)
        enc_input = dict(obs=batch['obs'], goal=batch['goal'])
        if use_contrastive_loss:
            enc_input.update(goal_aug=batch['goal_aug'])
        enc_dict = self.get_enc(enc_input)
        pred_ac = self._get_action(enc_dict['obs'], enc_dict['goal'])
        
        ret = dict(loss=None)
        # loss_action for comparison purposes is always MSE, but bc loss could for example be HuberLoss
        loss_action = nn.MSELoss()(pred_ac, act)
        loss = self.bc_loss(pred_ac, act)
        ret.update(loss_action=loss_action.detach(), pred_ac=pred_ac.detach())
        
        # aux. loss: prediction of target_color based on goal enc
        if self.conf.get('use_target_color_loss', False):
            pred_color = self._get_target_color(enc_dict['goal'])
            loss_color = nn.MSELoss()(pred_color, batch['target_color'])
            ret.update(loss_color=loss_color.detach(), pred_color=pred_color.detach())
            loss += self.conf.use_target_color_loss * loss_color
        
        # aux. loss: prediction of target_eef position based on goal and obs enc.
        if self.conf.get('use_target_eef_loss', False):
            pred_eef = self._get_eef(enc_dict['obs'][:, -1], enc_dict['goal'])
            loss_eef = nn.MSELoss()(pred_eef, batch['target_eef'])
            ret.update(loss_eef=loss_eef.detach(), pred_eef=pred_eef.detach())
            loss += self.conf.use_target_eef_loss * loss_eef

        # aux. loss: contrastive loss for supervising the goal embedding (goal and goal_aug)
        if use_contrastive_loss:
            contra_loss = self.contra_loss(enc_dict['goal'], enc_dict['goal_aug'])
            ret.update(loss_contra=contra_loss.detach())
            loss += use_contrastive_loss * contra_loss

        if compute_loss:
            ret.update(loss=loss)
        
        # if self.training:
        #     import matplotlib.pyplot as plt
        #     def plot(idx):
        #         _, axes = plt.subplots(1, self.n_stack+1)
        #         axes = axes.flatten()
        #         for i in range(self.n_stack):
        #             axes[i].imshow(obs[idx, i*3:i*3+3].detach().cpu().permute(1,2,0))
        #         axes[-1].imshow(goal[idx].detach().cpu().permute(1,2,0))
        #         plt.tight_layout()
        #         plt.savefig('in_training_eef.png')
        #     plot(0)
        #     breakpoint()
        
        return ret

    def get_enc(self, batch):
        obs = batch['obs']
        goal = batch['goal']
        self._check_obs_and_goal(obs, goal)

        B, C, H, W = obs.shape

        if self.is_obs_image:
            y = [obs[:, i*3:i*3+3] for i in range(self.n_stack)]
            x = einops.rearrange(y, 'i B C H W -> (B i) C H W').contiguous()
            obs_emb = self.enc(x).view(B, self.n_stack, -1) # B, N, D
        else:
            obs_emb = obs

        ret = dict(obs=obs_emb)
        # goal / goal augmented
        for key in batch:
            if key.startswith('goal'):
                ret[key] = self.enc(batch[key].contiguous()) if self.is_goal_image else batch[key]
        return ret

    def get_action(self, obs, goal):
        B, C, H, W = obs.size()
        enc_dict = self.get_enc(dict(obs=obs, goal=goal))
        pred_ac = self._get_action(enc_dict['obs'], enc_dict['goal'])
        return dict(action=pred_ac)

    def log_ret_dict(self, dictionary, step: str = 'train', is_batch: bool = True, **kwargs) -> None:
        suf = 'batch' if is_batch else 'epoch'
        for key in dictionary:
            if key.startswith('loss'):
                prog_bar = not is_batch and key == 'loss'
                self.log(f'{step}_{key}_{suf}', dictionary[key], prog_bar=prog_bar ,**kwargs)

    def log_ret_epoch(self, outputs, step: str = 'train', **kwargs):
        metric_vals = defaultdict(list)
        for output in outputs:
            for key in output:
                if key.startswith('loss'):
                    metric_vals[key].append(output[key])
        metric_vals = {k: torch.stack(v, 0).mean() for k, v in metric_vals.items()}
        self.log_ret_dict(metric_vals, step, is_batch=False, **kwargs)

    def training_step(self, batch):
        ret = self.ff(batch[0], compute_loss=True)
        self.log_ret_dict(ret, step='train', is_batch=True)
        return ret
        
    def training_epoch_end(self, outputs) -> None:
        self.log_ret_epoch(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        ret = self.ff(batch, compute_loss=True)
        return ret

    def validation_epoch_end(self, outputs) -> None:
        self.log_ret_epoch(outputs, 'valid')

    def _get_action(self, obs_emb, goal_emb):
        B, N, D = obs_emb.size()
        mlp_in = torch.cat([obs_emb.view(B, -1).contiguous(), goal_emb], -1)
        pred_ac = self.ctrl_net(mlp_in)
        pred_ac = pred_ac.tanh()

        return pred_ac

    def _get_eef(self, obs_emb, goal_emb):
        B, D = obs_emb.size()
        mlp_in = torch.cat([obs_emb, goal_emb], -1)
        pred_eef = self.eef_net(mlp_in)
        pred_eef = pred_eef.tanh()
        return pred_eef

    def _get_target_color(self, goal_emb):
        pred_colors = self.color_net(goal_emb)
        pred_colors = pred_colors.tanh()
        return pred_colors



class Reacher2dCtrlNet(nn.Module):

    def __init__(self, obs_shape, hidden_dim, img_enc_dim, n_stack) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.img_enc = img_enc_dim
        self.n_stack = n_stack
        self.obs_shape = obs_shape
        self.enc = EncoderWithBnorm(obs_shape=self.obs_shape, h_dim=self.img_enc)
        self.eef_network = MLP(
            in_channel=2 + self.n_stack * self.img_enc, 
            out_channel=2,
            hidden_dim=self.hidden_dim , 
            n_layers=3
        )


    def forward(self, batch, compute_loss=True):
        
        obs = batch['obs']
        B = obs.shape[0]
        y = [obs[:, i*3:i*3+3] for i in range(self.n_stack)]
        x = einops.rearrange(y, 'i B C H W -> (B i) C H W').contiguous()
        obs_emb = self.enc(x).view(B, self.n_stack, -1) # B, N, D

        eef_in = torch.cat([batch['target_eef'], obs_emb.view(B, -1)], -1)
        pred_ac = self.eef_network(eef_in)
        ret = dict(pred_ac=pred_ac.detach())
        if compute_loss:
            loss = nn.MSELoss()(pred_ac, batch['action'])
            ret['loss'] = loss
        return ret

    def get_action(self, obs, target_eef):
        batch = {'target_eef': target_eef, 'obs': obs}
        ret = self(batch, compute_loss=False)
        return dict(action=ret['pred_ac'])

    
class GCBCv6(GCBCv5):

    enc_cls = {
        'normal': Encoder,
        'resnet': EncoderResNet,
        'normal_bnorm': EncoderWithBnorm,
    }
    def __init__(self, conf):
        super().__init__(conf)

    def _check_obs_and_goal(self, obs, goal):
        assert obs.shape[1:] == self.conf.obs_shape, 'observation shape is not correct.'
        assert goal.shape[1:] == self.conf.goal_shape, 'goal shape is not correct.'

    def _build_network(self):
        super()._build_network()
        # we already should have an image encoder object due to super()
        # this will output the eef xy location
        self.eef_net = MLP(2 * self.conf.goal_enc_dim, 2, self.conf.hidden_dim, 3)
        # this will map eef to torque actions
        self.ctrl_net = Reacher2dCtrlNet(
            (3, ) + self.conf.obs_shape[1:],  # C=3, H, W
            self.conf.hidden_dim,
            self.conf.goal_enc_dim,
            self.n_stack
        )


    def bc_loss(self, pred_ac, target_ac):
        pred_ac = pred_ac.view(-1, pred_ac.shape[-1])
        target_ac = target_ac.view(-1, target_ac.shape[-1])

        if self.conf.get('use_huber_loss', False):
            loss = nn.HuberLoss(delta=0.05)(pred_ac, target_ac)
        else:
            loss = nn.MSELoss()(pred_ac, target_ac)
        
        return loss


    def ff(self, batch, compute_loss=True):
        act = batch['action']
        eef = batch['target_eef']
        
        output = self.get_action(batch['obs'], batch['goal'])
        pred_eef, pred_ac = output['eef'], output['action']
        
        ret = dict(loss=None)
        # loss_action for comparison purposes is always MSE, but bc loss could for example be HuberLoss
        loss_eef = nn.MSELoss()(pred_eef, eef)
        loss_action = nn.MSELoss()(pred_ac, act) # only for logging
        loss = loss_eef

        ret.update(
            loss_eef=loss_eef.detach(), pred_eef=pred_eef.detach(),
            loss_action=loss_action.detach(), pred_ac=pred_ac.detach(),
        )

        if compute_loss:
            ret.update(loss=loss)
        
        return ret

    def get_action(self, obs, goal):
        B, C, H, W = obs.size()
                
        goal_emb = self.enc(goal.contiguous())
        obs_emb = self.enc(obs[:, -3:].contiguous()) # only emb the last time step for eef prediction

        pred_eef = self._get_eef(obs_emb, goal_emb)
        pred_ac = self._get_action(obs, pred_eef)
        return dict(action=pred_ac, eef=pred_eef)

    def _get_eef(self, obs_emb, goal_emb, apply_tanh=True):
        mlp_in = torch.cat([obs_emb, goal_emb], -1)
        pred_eef = self.eef_net(mlp_in)
        if apply_tanh:
            pred_eef = pred_eef.tanh()

        return pred_eef

    def _get_action(self, obs_stack, target_eef):
        pred_ac = self.ctrl_net.get_action(obs_stack, target_eef)['action']
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
            # reserve one extra token for cls which will be added in the model
            max_position_embeddings=self.conf.max_ep_len + 1,
        )

        self.transformer = BertModel(bert_config)


    def forward(self, states, actions, attn_masks):
        # states: B, T, d_s
        # actions: B, T, d_a
        # masks: B, T (make sure it always includes first and last timesteps)

        B, T, _ = states.size()
        timesteps = torch.arange(T).to(device=states.device, dtype=torch.long)

        state_emb_in = self.emb_state(states) + self.emb_timestep(timesteps)
        action_emb_in = self.emb_action(actions) + self.emb_timestep(timesteps)

        input_tokens = einops.rearrange([state_emb_in, action_emb_in], 'i B T h -> B (T i) h')
        mask_tokens = einops.rearrange([attn_masks, attn_masks], 'i B T -> B (T i)')

        input_tokens = torch.cat([self.cls_token.repeat(B, 1, 1), input_tokens], dim=1)
        mask_tokens = torch.cat([torch.ones(B, 1, dtype=torch.bool).to(mask_tokens), mask_tokens], dim=1)

        tf_outputs = self.transformer(inputs_embeds=input_tokens, attention_mask=mask_tokens)
        # traj_emb = tf_outputs['pooler_output']

        return tf_outputs



class GCTrajGPT(nn.Module):
    """GPT style policy as the decoder"""

    def __init__(self, conf):
        super().__init__()

        conf = utils.ParamDict(conf)
        self.conf = conf
        self.h_dim = conf.hidden_dim
        self.obs_shape = conf.obs_shape
        self.ac_dim = conf.ac_dim
        # self.n_layers = conf.n_layers
        self.goal_dim = conf.goal_dim

        self.emb_timestep = None
        self.emb_state, self.emb_action = None, None
        self.gpt: Optional[GPT2Model] = None
        self._build_network()

    def _build_network(self):
        """ builds emb_state, emb_action, emb_timestep, transformer networks """
        self.emb_timestep = nn.Embedding(self.conf.max_ep_len, self.conf.hidden_dim)

        obs_shape = self.obs_shape
        h_dim = self.h_dim
        ac_dim = self.ac_dim
        goal_dim = self.goal_dim

        if len(obs_shape) != 1:
            raise ValueError('obs_shape should have one dimension.')

        self.emb_state = nn.Linear(obs_shape[0] + goal_dim, h_dim)
        self.emb_action = nn.Linear(ac_dim + goal_dim, h_dim)

        gpt_config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=h_dim,
            n_layer=3,
            n_head=1,
            n_inner=h_dim,
            # reserve one extra token for cls which will be added in the model
            n_positions=self.conf.max_ep_len,
        )

        self.gpt = GPT2Model(gpt_config)

        self.output_ac_nn = nn.Linear(h_dim, ac_dim)

    def bc_loss(self, pred_ac, target_ac):
     
        pred_ac = pred_ac.view(-1, pred_ac.shape[-1])
        target_ac = target_ac.view(-1, target_ac.shape[-1])
        loss = F.mse_loss(pred_ac, target_ac)
        
        return loss

    def forward(self, states, actions, attn_masks, goals):
        # states: B, T, d_s
        # actions: B, T, d_a
        # goals: B, d_g
        # masks: B, T (it includes the zero padded trajectories that are shorter)

        B, T, _ = states.size()
        timesteps = torch.arange(T).to(device=states.device, dtype=torch.long)

        goal_broadcasted = goals.unsqueeze(1).repeat(1, T, 1)
        state_goal_input = torch.cat([states, goal_broadcasted], -1)
        action_goal_input = torch.cat([actions, goal_broadcasted], -1)

        state_emb_in = self.emb_state(state_goal_input) + self.emb_timestep(timesteps)
        action_emb_in = self.emb_action(action_goal_input) + self.emb_timestep(timesteps)

        input_tokens = einops.rearrange([state_emb_in, action_emb_in], 'i B T h -> B (T i) h')
        mask_tokens = einops.rearrange([attn_masks, attn_masks], 'i B T -> B (T i)')

        tf_outputs = self.gpt(inputs_embeds=input_tokens, attention_mask=mask_tokens)

        return tf_outputs

    def get_state_action_embs(self, states, actions, attn_mask, goals):
        tf_outputs = self(states, actions, attn_mask, goals)
        output_tokens = tf_outputs['last_hidden_state']
        state_emb, action_emb = output_tokens[:, ::2, :], output_tokens[:, 1::2, :]
        return state_emb, action_emb

    def ff(self, states, actions, attn_mask, goals, compute_loss=True):
        state_emb, action_emb = self.get_state_action_embs(states, actions, attn_mask, goals)
        
        pred_ac = self.output_ac_nn(state_emb)

        ret = dict(pred_ac=pred_ac)
        if compute_loss:
            loss = self.bc_loss(pred_ac[attn_mask.bool()], actions[attn_mask.bool()])
            ret['loss']=loss

        return ret

    def get_action(self, cur_past_states, goal, past_actions=None):
        # cur_past_states: [T, d_s]
        # past_actions: [T-1, d_a]
        # goal: [d_g]
        # return action: [d_a]

        # add a dummy action, make a ff pass 
        # and just use the last state emb (which won't see the dummy anyways)

        T, _ = cur_past_states.size()

        if past_actions is None:
            actions = torch.ones(1, self.ac_dim).to(cur_past_states)
        else:
            assert past_actions.shape[0] == T - 1
            actions = torch.cat([past_actions, torch.ones(1, self.ac_dim).to(past_actions)], 0)
        
        states = cur_past_states[None]
        actions = actions[None]
        goals = goal[None]
        attn_mask = torch.ones(1, T).to(cur_past_states).long()

        output_dict = self.ff(states, actions, attn_mask, goals, compute_loss=False)
        next_action = output_dict['pred_ac'][0, -1] # take b=0, and t=T
        return next_action



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



class TOsilv1(BaseLightningModule):

    def _build_network(self):
        enc_config = utils.ParamDict(
            hidden_dim=self.conf.hidden_dim,
            obs_shape=(self.conf.obs_dim, ),
            ac_dim=self.conf.ac_dim,
            max_ep_len=self.conf.max_ep_len,
        )
        self.encoder = TrajBERT(enc_config)
        self.goal_dim = self.conf.goal_dim
        self.goal_emb = nn.Linear(self.conf.hidden_dim, self.goal_dim)


        self.use_gpt_decoder = self.conf.get('use_gpt_decoder', False)
        if self.use_gpt_decoder:
            dec_config = utils.ParamDict(
                hidden_dim=self.conf.hidden_dim,
                obs_shape=(self.conf.obs_dim, ),
                ac_dim=self.conf.ac_dim,
                goal_dim=self.conf.goal_dim,
                max_ep_len=self.conf.max_ep_len,
            )
            self.decoder = GCTrajGPT(dec_config)
        else:
            dec_config = utils.ParamDict(
                hidden_dim=self.conf.hidden_dim,
                obs_dim=self.conf.obs_dim,
                ac_dim=self.conf.ac_dim,
                goal_dim=self.conf.goal_dim,
            )

            self.decoder = GCBCv2(dec_config)

    def get_task_emb(self, context_s, context_a, context_mask):
        enc_output = self.encoder(context_s, context_a, context_mask)
        hstate = enc_output.last_hidden_state[:, 0]
        task_emb = self.goal_emb(hstate)
        return task_emb

    def ff(self, batch, compute_loss=True):
        context_s       = batch['context_s']
        context_a       = batch['context_a']
        context_mask    = batch['attention_mask']

        #### TODO: remove
        if 'context_class_id' in batch:
            class_acc = (batch['context_class_id'] == batch['target_class_id']).float().mean()
            self.log('class_acc', class_acc, prog_bar=True)

        # shape B,
        task_emb = self.get_task_emb(context_s, context_a, context_mask)

        if self.use_gpt_decoder:
            # shape B, T
            target_s    = batch['target_s']
            target_a    = batch['target_a']
            target_mask = batch['target_mask']
            ret = self.decoder.ff(target_s, target_a, target_mask, task_emb, compute_loss=compute_loss)
        else:
            target_s    = batch['target_s']
            target_a    = batch['target_a']
            ptr         = batch['ptr']

            task_emb_broadcasted = torch.repeat_interleave(task_emb, ptr, dim=0)
            ret = self.decoder.ff((target_s, task_emb_broadcasted, target_a), compute_loss=compute_loss)
        return ret



@dataclasses.dataclass
class CUBE:
    ind: int
    pos: slice
    color: slice

@dataclasses.dataclass
class EEF:
    pos: slice

def get_goal_color(state):
    # accepts batched input
    # inds of state that correspond to each cube
    CUBES = [
        CUBE(0, slice(8, 10), slice(10, 13)),
        CUBE(1, slice(13, 15), slice(15, 18)),
        CUBE(2, slice(18, 20), slice(20, 23)),
    ]
    eef = EEF(slice(4, 6))

    cube_colors = torch.stack([state[:, cube.color] for cube in CUBES], 1)
    cube_poses = torch.stack([state[:, cube.pos] for cube in CUBES], 1)
    eef_pos = state[:, eef.pos][:, None, :]
    
    dists = ((cube_poses - eef_pos)**2).sum(-1)
    target_cube_ind = dists.argmin(-1)
    target_color = cube_colors[torch.arange(eef_pos.shape[0]), target_cube_ind]
    info = {
        'eef_pos': eef_pos,
        'cube_poses': cube_poses,
        'cube_colors': cube_colors,
        'dists': dists,
        'argmin': target_cube_ind,
        'color': target_color,
    }
    return target_color, info

class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # z_i = emb_i
        # z_j = emb_j

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss


        
class TOsilv1DebugReacher(BaseLightningModule):

    def _build_network(self):
        # enc_config = utils.ParamDict(
        #     hidden_dim=self.conf.hidden_dim,
        #     obs_shape=(self.conf.obs_dim, ),
        #     ac_dim=self.conf.ac_dim,
        #     max_ep_len=self.conf.max_ep_len,
        # )
        # self.encoder = TrajBERT(enc_config)
        # self.goal_dim = self.conf.goal_dim
        # self.goal_emb = nn.Linear(self.conf.hidden_dim, self.goal_dim)

        obs_dim = self.conf.obs_dim
        h_dim = self.conf.hidden_dim
        goal_dim = self.conf.goal_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, goal_dim)
        )

        self.use_gpt_decoder = self.conf.get('use_gpt_decoder', False)
        if self.use_gpt_decoder:
            dec_config = utils.ParamDict(
                hidden_dim=self.conf.hidden_dim,
                obs_shape=(self.conf.obs_dim, ),
                ac_dim=self.conf.ac_dim,
                goal_dim=self.conf.goal_dim,
                max_ep_len=self.conf.max_ep_len,
            )
            self.decoder = GCTrajGPT(dec_config)
        else:
            dec_config = utils.ParamDict(
                hidden_dim=self.conf.hidden_dim,
                obs_dim=self.conf.obs_dim,
                ac_dim=self.conf.ac_dim,
                goal_dim=self.conf.goal_dim,
            )

            self.decoder = GCBCv2(dec_config)

        self.grid2spher_proj = MLP(3, goal_dim, 128, 3, nn.Tanh())


    def get_task_emb(self, context_s, context_a, context_mask):
        # enc_output = self.encoder(context_s, context_a, context_mask)
        # hstate = enc_output.last_hidden_state[:, 0]
        # task_emb = self.goal_emb(hstate)
        # B = context_s.shape[0]
        # inds = context_mask.sum(-1) - 1
        last_states = context_s[:, -1]
        task_emb = self.encoder(last_states)
        return task_emb

    def ff(self, batch, compute_loss=True):
        context_s       = batch['context_s']
        context_a       = batch['context_a']
        context_mask    = batch['attention_mask']

        B = context_s.shape[0]

        #### TODO: remove
        if 'context_class_id' in batch:
            class_acc = (batch['context_class_id'] == batch['target_class_id']).float().mean()
            self.log('class_acc', class_acc, prog_bar=True)

        # shape B,
        # task_emb = self.get_task_emb(context_s, context_a, context_mask)
        task_goal_color, _ = get_goal_color(context_s[:, -1])

        color_vector = self.grid2spher_proj(task_goal_color.detach())
        color_vector = F.normalize(color_vector, -1)

        loss_proj = ContrastiveLossELI5(B)(color_vector, color_vector)

        # (B, 1, D) x (B, D, 1) = (B, 1, 1)
        # inner_prod = torch.bmm(color_vector.unsqueeze(1), task_emb.unsqueeze(-1))
        # loss_fit = -inner_prod.mean()


        # try to fit the goal color directly / contrastively
        # target_s_enc    = batch['target_s_enc']
        # target_a_enc    = batch['target_a_enc']
        # target_mask_enc = batch['target_mask_enc']
        # target_emb = self.get_task_emb(target_s_enc, target_a_enc, target_mask_enc)

        # target_goal_color, _ = get_goal_color(target_s_enc[:, -1])
        # similar_flag = torch.all(task_goal_color.repeat(B, 1, 1).transpose(0, 1) == target_goal_color.repeat(B, 1, 1), -1)

        # task_emb_norm = task_emb / (task_emb**2).sum(-1, keepdim=True)**0.5
        # target_emb_norm = target_emb / (target_emb**2).sum(-1, keepdim=True)**0.5
        # # task_emb_norm = task_emb 
        # # target_emb_norm = target_emb
        # # target_emb_norm = target_emb_norm.detach() # detach to stabilize training
        # logits = task_emb_norm @ target_emb_norm.T
        # logits = (logits - logits.mean(-1, keepdim=True)) / logits.std(-1, keepdim=True)
        # # contrastive_logits = contrastive_logits - contrastive_logits.max(-1, keepdim=True)[0] # subtract the mean
        # # contrastive_logits = contrastive_logits - contrastive_logits.mean(-1, keepdim=True) # subtract the mean
        
        # loss_colors = nn.BCEWithLogitsLoss()(logits, similar_flag.float())
        # target_emb = target_emb.detach()
        # cosine similarity
        # task_emb_norm   = F.normalize(task_emb, dim=-1)
        # target_emb_norm = F.normalize(target_emb, dim=-1)
        # l1 = task_emb_norm @ target_emb_norm.T
        # l2 = target_emb_norm @ task_emb_norm.T
        # l1 -= l1.mean(-1, keepdim=True)
        # l2 -= l2.mean(-1, keepdim=True)
        # euclidean similarity
        # contrastive_dist = task_emb.repeat(B, 1, 1).transpose(0, 1) - target_emb.repeat(B, 1, 1).detach()
        # contrastive_logits = -(contrastive_dist**2).sum(-1)**0.5
        # contrastive_logits = contrastive_logits - contrastive_logits.max(-1, keepdim=True)[0]
        # divide by a temprature norm for sharpening
        # contrastive_logits /= 0.1

        # labels = torch.arange(B).to(task_emb).long()
        # loss_colors = nn.CrossEntropyLoss()(contrastive_logits, labels)
        # loss_colors = 0.5 * (nn.CrossEntropyLoss()(l1, labels) + nn.CrossEntropyLoss()(l2, labels))
        # goal_colors, _ = get_goal_color(context_s[:, -1])
        # loss_colors = nn.MSELoss()(goal_colors, task_emb)
        # loss_colors = ContrastiveLossELI5(B)(task_emb, )
        # loss_colors = ContrastiveLossELI5(B)(task_emb, task_emb)

        # if self.use_gpt_decoder:
        #     # shape B, T
        #     target_s    = batch['target_s']
        #     target_a    = batch['target_a']
        #     target_mask = batch['target_mask']
        #     ret = self.decoder.ff(target_s, target_a, target_mask, task_emb, compute_loss=compute_loss)
        # else:
        #     target_s    = batch['target_s']
        #     target_a    = batch['target_a']
        #     ptr         = batch['ptr']

        #     task_emb_broadcasted = torch.repeat_interleave(task_emb, ptr, dim=0)
        #     ret = self.decoder.ff((target_s, task_emb_broadcasted, target_a), compute_loss=compute_loss)
        
        ret = {}
        # ret['loss_bc'] = ret['loss'].clone()
        # ret['loss_colors'] = loss_colors
        # ret['loss'] += loss_colors
        # ret['loss'] = loss_colors
        ret['loss'] = loss_proj
        return ret


class TOsilSemisupervised(TOsilv1):

    def _init_conf(self, conf):
        super()._init_conf(conf)
        self.decoder_loss_weight = self.conf.decoder_loss_weight
        self.mse = nn.MSELoss()

    def _build_network(self):
        super()._build_network()
        self._state_mask_token = nn.Parameter(torch.randn(self.conf.obs_dim), requires_grad=True)
        self._action_mask_token = nn.Parameter(torch.randn(self.conf.ac_dim), requires_grad=True)
        self.state_output_proj = nn.Linear(self.conf.hidden_dim, self.conf.obs_dim)
        self.action_output_proj = nn.Linear(self.conf.hidden_dim, self.conf.ac_dim)

    def _mask_input(self, states, actions, attn_mask):
        """
        We chose to mask states and actions identically
        TODO: should we introduce the masked token in the embedding space or in the input space?
        """
        masked_states = states.clone()
        masked_actions = actions.clone()

        # mask those that are not padded (attm_mask = 0)
        B, T, _ = states.size()
        mask_rate = self.conf.mask_rate
        # mask p percent of the input tokens for each batch
        # mask = 0 means no attention
        mask_canvas = torch.zeros(B, T).long().to(device=states.device)
        total_steps = attn_mask.sum()
        masked_time_steps = (torch.rand(total_steps, device=states.device) < mask_rate).long()
        mask_canvas[attn_mask.bool()] = masked_time_steps

        # make sure the first and last timesteps are part of the input seq
        mask_canvas[:, 0] = 0
        mask_canvas[torch.arange(B), attn_mask.sum(-1) - 1] = 0

        masked_states[mask_canvas.bool()] = self._state_mask_token.repeat(mask_canvas.sum(), 1)
        masked_actions[mask_canvas.bool()] = self._action_mask_token.repeat(mask_canvas.sum(), 1)

        return masked_states, masked_actions, mask_canvas, mask_canvas

    def ff_encoder(self, batch, compute_loss=True):
        states       = batch['context_s']
        actions      = batch['context_a']
        attn_mask    = batch['attention_mask']

        masked_states, masked_actions, masked_state_inds, masked_action_inds = self._mask_input(states, actions, attn_mask)
        encoder_output = self.encoder(masked_states, masked_actions, attn_mask)
        
        # the first token is cls then it's a state and then it's an action
        state_embs = encoder_output.last_hidden_state[:, 1::2]
        action_embs = encoder_output.last_hidden_state[:, 2::2]

        # pass the masked embeddings through a prediction head
        predicted_states = self.state_output_proj(state_embs[masked_state_inds.bool()])
        predicted_actions = self.action_output_proj(action_embs[masked_action_inds.bool()])

        ret = dict(encoder_output=encoder_output)
        if compute_loss:        
            target_states = states[masked_state_inds.bool()]
            target_actions = actions[masked_action_inds.bool()]

            # compute individual action and state prediction errors and sum them up
            loss_actions = self.mse(target_actions, predicted_actions)
            loss_states = self.mse(target_states, predicted_states)
            loss = loss_actions + loss_states  

            ret.update(
                loss=loss,
                loss_actions=loss_actions,
                loss_states=loss_states
            )

        return ret

    def ff_decoder(self, batch, compute_loss=True):
        context_s       = batch['context_s']
        context_a       = batch['context_a']
        context_mask    = batch['attention_mask']

        task_emb = self.get_task_emb(context_s, context_a, context_mask)

        target_s    = batch['target_s']
        target_a    = batch['target_a']
        ptr         = batch['ptr']

        task_emb_broadcasted = torch.repeat_interleave(task_emb, ptr, dim=0)
        ret = self.decoder.ff((target_s, task_emb_broadcasted, target_a), compute_loss=compute_loss)
        return ret

    def ff(self, batch, compute_loss=True):
        # compute the Masked trajectory modeling loss
        encoder_loss = 0
        if 'unpaired' in batch:
            encoder_output = self.ff_encoder(batch['unpaired'], compute_loss=compute_loss)
            encoder_loss = encoder_output['loss']

        # compute the loss from paired trajectories
        decoder_loss = 0
        if 'paired' in batch:
            decoder_output = self.ff_decoder(batch['paired'], compute_loss=compute_loss)
            decoder_loss = decoder_output['loss']
        
        # compute the total training loss
        loss = (1 - self.decoder_loss_weight) * encoder_loss + self.decoder_loss_weight * decoder_loss
        
        return dict(loss=loss, encoder_loss=encoder_loss.detach(), decoder_loss=decoder_loss.detach())

    def training_step(self, batch, batch_idx):
        ret = self.ff(batch, compute_loss=True)
        self.log('train_loss_batch', ret['loss'])
        return ret

    def training_epoch_end(self, outputs) -> None:
        train_losses = torch.stack([item['loss'] for item in outputs], 0)
        train_enc_losses = torch.stack([item['encoder_loss'] for item in outputs], 0)
        train_dec_losses = torch.stack([item['decoder_loss'] for item in outputs], 0)
        self.log('train_loss_epoch', train_losses.mean(), prog_bar=True)
        self.log('train_enc_loss_epoch', train_enc_losses.mean(), prog_bar=False)
        self.log('train_dec_loss_epoch', train_dec_losses.mean(), prog_bar=False)

    def validation_step(self, batch, batch_idx):
        batch = dict(paired=batch, unpaired=batch)
        ret = self.ff(batch, compute_loss=True)
        return ret

    def validation_epoch_end(self, outputs) -> None:
        valid_losses = torch.stack([item['loss'] for item in outputs], 0)
        valid_enc_losses = torch.stack([item['encoder_loss'] for item in outputs], 0)
        valid_dec_losses = torch.stack([item['decoder_loss'] for item in outputs], 0)
        # this should be compared to training loss to monitor overfitting
        self.log('valid_loss_epoch', valid_losses.mean(), prog_bar=True)
        # to keep things consistent with other models valid_loss should 
        # measure the decoding capability
        self.log('valid_loss', valid_dec_losses.mean())
        self.log('valid_enc_loss_epoch', valid_enc_losses.mean())
        self.log('valid_dec_loss_epoch', valid_dec_losses.mean())

