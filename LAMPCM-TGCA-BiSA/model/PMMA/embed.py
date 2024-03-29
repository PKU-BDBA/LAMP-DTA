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


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, mol_len, in_channels=3):
        super(Embeddings, self).__init__()
        num_mol = mol_len

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mol_embeddings = Linear(config.hidden_size, config.hidden_size)
                        
        self.position_embedding = nn.Parameter(torch.zeros(1, config.feat_len + 1, config.hidden_size)) # TODO
        self.pe_mol = nn.Parameter(torch.zeros(1, num_mol, config.hidden_size))

        self.dropout_mol = Dropout(config.transformer["dropout_rate"])
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, prot, mol):
        if mol != None:
            mol = self.mol_embeddings(mol)
            mol_embeddings = mol + self.pe_mol
            mol_embeddings = self.dropout_mol(mol_embeddings)
        else:
            mol_embeddings=None

        N = prot.shape[0]
        cls_tokens = self.cls_token.expand(N, -1, -1)
        prot = torch.cat((cls_tokens, prot), dim=1)

        embeddings = prot + self.position_embedding
        embeddings = self.dropout(embeddings)

        return embeddings, mol_embeddings


