# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import model.PMMA.configs as configs
from model.PMMA.attention import Attention
from model.PMMA.embed import Embeddings 
from model.PMMA.mlp import Mlp
from model.PMMA.block import PMMABlock

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer_with_mol = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.embed_dim, eps=1e-6)
        for i in range(config.transformer["num_p_plus_s_layers"]):
            if i < 2:
                layer_with_mol = PMMABlock(config, vis, mm=True)
            else:
                layer_with_mol = PMMABlock(config, vis)
            self.layer_with_mol.append(copy.deepcopy(layer_with_mol))
            
        for i in range(config.transformer["num_p_plus_s_layers"]):
            layer_with_mol = PMMABlock(config, vis)

    def forward(self, hidden_states, mol=None):
        attn_weights = []
        guided_attn_weights = []
        
        for (i, layer_block) in enumerate(self.layer_with_mol):
            if i>=2:
                hidden_states = torch.cat((hidden_states, mol), 1) 
                hidden_states, weights, guided_weights = layer_block(hidden_states)
            else:
                hidden_states, mol, weights, guided_weights = layer_block(hidden_states, mol)
            if self.vis:
                attn_weights.append(weights)
                guided_attn_weights.append(guided_weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights, guided_attn_weights

